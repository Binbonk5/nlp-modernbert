from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm

# ---------------------------------------------------------
# 1. CẤU HÌNH HỆ THỐNG PHÂN LẬP TUYỆT ĐỐI (CRANFIELD MINI)
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATASET_NAME = "cranfield"  # Đã đổi thành bộ Cranfield mini
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_SLUG = MODEL_NAME.replace("/", "_")
RESULTS_FILE = RESULTS_DIR / f"cranfield_modernbert_large_{MODEL_SLUG}.json"

QUERY_BATCH_SIZE = 32
CORPUS_BATCH_SIZE = 64
CORPUS_CHUNK_SIZE = 1024
TOP_K = 10
MAX_SEQ_LENGTH = 512


def build_model(model_name: str, device: str):
    transformer = models.Transformer(
        model_name,
        model_args={
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2",
        },
        tokenizer_args={"use_fast": True},
    )
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    model = SentenceTransformer(modules=[transformer, pooling], device=device)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def format_passage(doc: Dict[str, str]) -> str:
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    if title and text:
        return f"{title} {text}"
    return title or text


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int):
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


@torch.no_grad()
def retrieve_topk(
    model: SentenceTransformer,
    corpus: Dict[str, Dict[str, str]],
    queries: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    device = torch.device(model.device)
    corpus_ids = list(corpus.keys())
    corpus_texts = [format_passage(corpus[doc_id]) for doc_id in corpus_ids]
    query_ids = list(queries.keys())
    query_texts = list(queries.values())

    query_embeddings = encode_texts(model, query_texts, batch_size=QUERY_BATCH_SIZE).to(device)
    top_scores = torch.full((len(query_ids), TOP_K), float("-inf"), device=device)
    top_doc_idx = torch.full((len(query_ids), TOP_K), -1, dtype=torch.long, device=device)

    for start in tqdm(range(0, len(corpus_ids), CORPUS_CHUNK_SIZE), desc="Tiến độ xử lý Chunks", unit="chunk"):
        end = min(start + CORPUS_CHUNK_SIZE, len(corpus_ids))
        chunk_embeddings = encode_texts(model, corpus_texts[start:end], batch_size=CORPUS_BATCH_SIZE).to(device)
        scores = torch.matmul(query_embeddings, chunk_embeddings.transpose(0, 1))
        chunk_indices = torch.arange(start, end, device=device).unsqueeze(0).expand(scores.size(0), -1)

        combined_scores = torch.cat([top_scores, scores], dim=1)
        combined_indices = torch.cat([top_doc_idx, chunk_indices], dim=1)
        new_scores, new_positions = torch.topk(combined_scores, k=TOP_K, dim=1)
        new_indices = torch.gather(combined_indices, 1, new_positions)
        top_scores, top_doc_idx = new_scores, new_indices

    results = {}
    for row_index, query_id in enumerate(query_ids):
        results[query_id] = {
            corpus_ids[doc_index.item()]: top_scores[row_index, column_index].item()
            for column_index, doc_index in enumerate(top_doc_idx[row_index])
            if doc_index.item() >= 0
        }
    return results


def main():
    print(f"=== BẮT ĐẦU CHẠY THỰC NGHIỆM: {MODEL_NAME} ===")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[*] Thiết bị sử dụng : GPU - {torch.cuda.get_device_name(0)}")
    else:
        print("[!] CẢNH BÁO: Không nhận diện được GPU, đang chạy bằng CPU!")

    # ---------------------------------------------------------
    # ĐÃ THAY ĐỔI: Sử dụng ir_datasets để map cấu hình siêu nhẹ
    # ---------------------------------------------------------
    print("\n[1/5] Đang nạp tập dữ liệu Cranfield qua ir_datasets...")
    try:
        import ir_datasets
    except ImportError:
        raise ImportError("Vui lòng cài đặt thư viện ir_datasets bằng lệnh: pip install ir_datasets")

    raw_ir_dataset = ir_datasets.load("cranfield")

    # Ánh xạ cấu hình sang định dạng Dictionaries chuẩn BEIR
    corpus = {doc.doc_id: {"title": doc.title, "text": doc.text} for doc in raw_ir_dataset.docs_iter()}
    queries = {query.query_id: query.text for query in raw_ir_dataset.queries_iter()}
    
    qrels = {}
    for qrel in raw_ir_dataset.qrels_iter():
        if qrel.query_id not in qrels:
            qrels[qrel.query_id] = {}
        qrels[qrel.query_id][qrel.doc_id] = int(qrel.relevance)

    print(f"      -> Số lượng Document: {len(corpus)}")
    print(f"      -> Số lượng Query   : {len(queries)}")

    print("[2/5] Đang nạp mô hình vào bộ nhớ VRAM...")
    model = build_model(MODEL_NAME, device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print("[3/5] Bắt đầu quá trình mã hóa (Encoding) và tìm kiếm Vector...")
    start_time = time.perf_counter()
    results = retrieve_topk(model, corpus, queries)
    
    print("[4/5] Đang chấm điểm hiệu năng Retrieval hệ thống...")
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values=[TOP_K])
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    record = {
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "ndcg@10": round(float(ndcg[f"NDCG@{TOP_K}"]), 6),
        "recall@10": round(float(recall[f"Recall@{TOP_K}"]), 6),
        "time_sec": round(elapsed, 6),
        "mem_mb": round(float(peak_memory_mb), 2),
        "query_batch_size": QUERY_BATCH_SIZE,
        "corpus_batch_size": CORPUS_BATCH_SIZE,
        "corpus_chunk_size": CORPUS_CHUNK_SIZE,
        "top_k": TOP_K,
        "optimization": "flash_attention_2_sandbox_ready",
        "device": device,
        "results_file": str(RESULTS_FILE.relative_to(PROJECT_ROOT)),
        "corpus_size": len(corpus),
        "query_size": len(queries),
    }

    with RESULTS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\n=== KẾT QUẢ THỰC NGHIỆM CRANFIELD ===")
    print(json.dumps(record, indent=4))
    print(f"[*] Đã lưu file kết quả tại: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
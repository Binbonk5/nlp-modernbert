from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.util import download_and_unzip
from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATASET_NAME = "trec-covid"
# MODEL_NAME = "AnswerDotAI/ModernBERT-base"
MODEL_NAME = "joe32140/ModernBERT-base-msmarco"
MODEL_SLUG = MODEL_NAME.replace("/", "_")
RESULTS_FILE = RESULTS_DIR / f"trec_covid_modernbert_base_{MODEL_SLUG}.json"
DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"
QUERY_BATCH_SIZE = 32
CORPUS_BATCH_SIZE = 64
CORPUS_CHUNK_SIZE = 1024
TOP_K = 10
MAX_SEQ_LENGTH = 512


def build_model(model_name: str, device: str):
    transformer = models.Transformer(
        model_name,
        model_args={"torch_dtype": torch.bfloat16, "trust_remote_code": True, "attn_implementation": "flash_attention_2"},
        tokenizer_args={"use_fast": True},
    )
    pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode_mean_tokens=True, pooling_mode_cls_token=False, pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[transformer, pooling], device=device)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def format_passage(doc: Dict[str, str]) -> str:
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return f"{title} {text}".strip() if title and text else title or text


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int):
    return model.encode(texts, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)


@torch.no_grad()
def retrieve_topk(model: SentenceTransformer, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str]) -> Dict[str, Dict[str, float]]:
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
        top_scores = new_scores
        top_doc_idx = torch.gather(combined_indices, 1, new_positions)

    results = {}
    for row_index, query_id in enumerate(query_ids):
        results[query_id] = {
            corpus_ids[doc_index.item()]: top_scores[row_index, column_index].item()
            for column_index, doc_index in enumerate(top_doc_idx[row_index])
            if doc_index.item() >= 0
        }
    return results


def ensure_dataset(data_dir: Path) -> Path:
    data_folder = data_dir / DATASET_NAME
    if data_folder.exists():
        return data_folder
    data_folder.parent.mkdir(parents=True, exist_ok=True)
    extracted = download_and_unzip(DATASET_URL, str(data_folder.parent))
    return Path(extracted) if extracted else data_folder


def main():
    print(f"=== BẮT ĐẦU CHẠY THỰC NGHIỆM: {MODEL_NAME} ===")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[*] Thiết bị sử dụng : GPU - {torch.cuda.get_device_name(0)}")
    else:
        print("[!] CẢNH BÁO: Không nhận diện được GPU, đang chạy bằng CPU!")

    print("\n[1/5] Đang kiểm tra và tải dữ liệu...")
    data_folder = ensure_dataset(DATA_DIR)
    corpus, queries, qrels = GenericDataLoader(str(data_folder)).load(split="test")
    print(f"      -> Số lượng Document: {len(corpus)}")
    print(f"      -> Số lượng Query   : {len(queries)}")

    print("[2/5] Đang tải mô hình vào bộ nhớ...")
    model = build_model(MODEL_NAME, device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print("[3/5] Đang mã hóa câu hỏi (Queries)...")
    start_time = time.perf_counter()
    results = retrieve_topk(model, corpus, queries)
    evaluator = EvaluateRetrieval()
    print("[5/5] Đang chấm điểm so với đáp án (Qrels)...")
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values=[TOP_K])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    record = {
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "ndcg@10": round(float(ndcg[f"NDCG@{TOP_K}"]), 6),
        "recall@10": round(float(recall[f"Recall@{TOP_K}"]), 6),
        "time_sec": round(time.perf_counter() - start_time, 6),
        "mem_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
        "query_batch_size": QUERY_BATCH_SIZE,
        "corpus_batch_size": CORPUS_BATCH_SIZE,
        "corpus_chunk_size": CORPUS_CHUNK_SIZE,
        "top_k": TOP_K,
        "optimization": "flash_attention_2",
        "device": device,
        "results_file": str(RESULTS_FILE.relative_to(PROJECT_ROOT)),
        "corpus_size": len(corpus),
        "query_size": len(queries),
    }

    with RESULTS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\n=== KẾT QUẢ ===")
    print(json.dumps(record, indent=4))
    print(f"[*] Đã lưu kết quả tại: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
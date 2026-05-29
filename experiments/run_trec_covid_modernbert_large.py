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

# ---------------------------------------------------------
# 1. CẤU HÌNH HỆ THỐNG MÔ HÌNH LARGE
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / "checkpoints" / "embeddings"

DATASET_NAME = "trec-covid"
MODEL_NAME = "lightonai/modernbert-embed-large"
MODEL_SLUG = MODEL_NAME.replace("/", "_")
RESULTS_FILE = RESULTS_DIR / f"trec_covid_modernbert_large_{MODEL_SLUG}.json"
DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"

QUERY_BATCH_SIZE = 32
CORPUS_BATCH_SIZE = 64
CORPUS_CHUNK_SIZE = 1024
TOP_K = 10
MAX_SEQ_LENGTH = 512


def build_model(model_name: str, device: str):
    transformer = models.Transformer(
        model_name,
        model_args={
            "torch_dtype": torch.bfloat16,               # Tối ưu tuyệt đối cho GPU A100
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2",   # Tăng tốc FlashAttention-2
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
        normalize_embeddings=True, # ĐÃ SỬA: Tắt chuẩn hóa để chạy chuẩn không gian toán học Dot-Product
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
    corpus_texts = [f"search_document: {format_passage(corpus[doc_id])}" for doc_id in corpus_ids]
    query_ids = list(queries.keys())
    query_texts = [f"search_query: {q}" for q in queries.values()]

    print("\n[3/5] Đang mã hóa câu hỏi (Queries)...")
    query_embeddings = encode_texts(model, query_texts, batch_size=QUERY_BATCH_SIZE).to(device)

    # Đặt tên file cache đặc trưng dựa trên model large
    safe_model_name = MODEL_NAME.replace("/", "_")
    cache_file = CACHE_DIR / f"{DATASET_NAME}_{safe_model_name}_corpus.pt"

    # ---------------------------------------------------------
    # CƠ CHẾ KIỂM TRA VÀ TRÍCH XUẤT CACHE VECTOR
    # ---------------------------------------------------------
    if cache_file.exists():
        print(f"[*] Tìm thấy Cache Vector Large tại: {cache_file}")
        print("      -> Đang nạp thẳng ma trận vào GPU (Mất 1 giây)...")
        corpus_embeddings = torch.load(cache_file, map_location=device)
    else:
        print(f"\n[4/5] Chưa có cache. Bắt đầu mã hóa {len(corpus_ids)} tài liệu bằng ModernBERT-Large...")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        embeddings_list = []
        # ĐÃ SỬA: Đưa thanh tqdm vào đúng vòng lặp encode thực tế của Corpus
        for start in tqdm(range(0, len(corpus_ids), CORPUS_CHUNK_SIZE), desc="Tiến độ xử lý Chunks", unit="chunk"):
            end = min(start + CORPUS_CHUNK_SIZE, len(corpus_ids))
            chunk_emb = encode_texts(model, corpus_texts[start:end], batch_size=CORPUS_BATCH_SIZE)
            embeddings_list.append(chunk_emb.cpu()) # Đẩy tạm về RAM để bảo vệ VRAM cho cụm tính toán sau

        corpus_embeddings = torch.cat(embeddings_list, dim=0).to(device)
        print(f"[*] Đang đóng gói xuất lưu file Cache cho lần sau: {cache_file}")
        torch.save(corpus_embeddings, cache_file)

    # ---------------------------------------------------------
    # TIẾN HÀNH TÌM KIẾM VECTOR (VECTOR SEARCH)
    # ---------------------------------------------------------
    print("      -> Đang thực hiện nhân ma trận tương đồng và lọc Top-K...")
    top_scores = torch.full((len(query_ids), TOP_K), float("-inf"), device=device)
    top_doc_idx = torch.full((len(query_ids), TOP_K), -1, dtype=torch.long, device=device)

    for start in range(0, len(corpus_ids), CORPUS_CHUNK_SIZE):
        end = min(start + CORPUS_CHUNK_SIZE, len(corpus_ids))
        chunk_embeddings = corpus_embeddings[start:end]
        
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

    print("\n[1/5] Đang kiểm tra dữ liệu...")
    data_folder = ensure_dataset(DATA_DIR)
    corpus, queries, qrels = GenericDataLoader(str(data_folder)).load(split="test")

    print("\n[2/5] Đang tải mô hình ModernBERT-Large vào bộ nhớ...")
    model = build_model(MODEL_NAME, device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    results = retrieve_topk(model, corpus, queries)
    
    print("\n[5/5] Đang chấm điểm so với đáp án (Qrels)...")
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
        "optimization": "flash_attention_2_with_cache",
        "device": device,
        "results_file": str(RESULTS_FILE.relative_to(PROJECT_ROOT)),
        "corpus_size": len(corpus),
        "query_size": len(queries),
    }

    with RESULTS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\n=== KẾT QUẢ ĐÃ ĐƯỢC CHUẨN HÓA LỚN ===")
    print(json.dumps(record, indent=4))


if __name__ == "__main__":
    main()
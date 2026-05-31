from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict

import ir_datasets
import pandas as pd
import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer, util


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

DATASET_NAME = "cranfield"
MODEL_NAME = "lightonai/modernbert-embed-large"
MODEL_SLUG = MODEL_NAME.replace("/", "_")
RESULTS_FILE = RESULTS_DIR / f"cranfield_modernbert_large_{MODEL_SLUG}.json"
QUERY_BATCH_SIZE = 32
CORPUS_BATCH_SIZE = 64
TOP_K = 10
TOP_RETRIEVE = 100
MAX_SEQ_LENGTH = 512


def build_model(model_name: str, device: str) -> SentenceTransformer:
    model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def format_passage(doc: Dict[str, str]) -> str:
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    if title and text:
        return f"{title} {text}"
    return title or text


def load_cranfield():
    dataset = ir_datasets.load("cranfield")
    corpus = {
        doc.doc_id: {"title": doc.title, "text": doc.text}
        for doc in dataset.docs_iter()
    }
    queries = {query.query_id: query.text for query in dataset.queries_iter()}
    qrels: Dict[str, Dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)
    return corpus, queries, qrels


def evaluate_results(retrieval_results, qrels, model_name):
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, retrieval_results, [1, 5, 10])

    return {
        "model": model_name,
        "NDCG@10": round(float(ndcg["NDCG@10"]), 4),
        "MAP@10": round(float(_map["MAP@10"]), 4),
        "Recall@10": round(float(recall["Recall@10"]), 4),
        "Precision@10": round(float(precision["P@10"]), 4),
    }


def main():
    print("=== ĐANG CHẠY THỰC NGHIỆM MODERNBERT-EMBED-LARGE TRÊN CRANFIELD ===")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[*] Thiết bị sử dụng: GPU - {torch.cuda.get_device_name(0)}")
    else:
        print("[!] CẢNH BÁO: Không nhận diện được GPU, đang chạy bằng CPU!")

    print("\n[1/3] Đang tải dữ liệu Cranfield từ ir_datasets...")
    corpus, queries, qrels = load_cranfield()
    print(f"      -> Số lượng Document: {len(corpus)}")
    print(f"      -> Số lượng Query   : {len(queries)}")

    print("[2/3] Đang tải mô hình ModernBERT vào bộ nhớ...")
    model_modernbert = build_model(MODEL_NAME, device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    corpus_ids = list(corpus.keys())
    query_ids = list(queries.keys())
    corpus_texts = [format_passage(corpus[doc_id]) for doc_id in corpus_ids]
    query_texts = list(queries.values())

    print("[3/3] Đang mã hóa corpus và queries, rồi tính cosine similarity...")
    start_time = time.perf_counter()
    corpus_embeddings = model_modernbert.encode(
        corpus_texts,
        batch_size=CORPUS_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    query_embeddings = model_modernbert.encode(
        query_texts,
        batch_size=QUERY_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    cos_scores = util.cos_sim(query_embeddings, corpus_embeddings)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mem_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0

    modernbert_results = {}
    for row_index, query_id in enumerate(query_ids):
        top_results = torch.topk(cos_scores[row_index], k=min(TOP_RETRIEVE, len(corpus_ids)))
        modernbert_results[query_id] = {
            corpus_ids[idx]: float(score)
            for score, idx in zip(top_results.values, top_results.indices)
        }

    final_results = {
        MODEL_NAME: evaluate_results(modernbert_results, qrels, MODEL_NAME)
    }
    final_results[MODEL_NAME]["time_sec"] = round(time.perf_counter() - start_time, 6)
    final_results[MODEL_NAME]["device"] = device
    final_results[MODEL_NAME]["mem_mb"] = mem_mb
    final_results[MODEL_NAME]["corpus_size"] = len(corpus)
    final_results[MODEL_NAME]["query_size"] = len(queries)

    record = {
        "dataset": DATASET_NAME,
        "results": final_results[MODEL_NAME],
        "results_file": str(RESULTS_FILE.relative_to(PROJECT_ROOT)),
    }

    with RESULTS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\n=== BẢNG SO SÁNH KẾT QUẢ BENCHMARK TRÊN CRANFIELD ===")
    df_compare = pd.DataFrame.from_dict(final_results, orient="index")
    print(df_compare)
    print(f"[*] Đã lưu file kết quả tại: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
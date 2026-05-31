from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import ir_datasets
import numpy as np
import pandas as pd
import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

DATASET_NAME = "cranfield"
MODEL_NAME = "Okapi-BM25"
RESULTS_FILE = RESULTS_DIR / "cranfield_bm25.json"
TOP_K = 10
TOP_RETRIEVE = 100


def tokenize(text: str) -> List[str]:
	return text.lower().split()


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


def build_bm25_index(corpus_texts: List[str]) -> BM25Okapi:
	tokenized_corpus = [tokenize(text) for text in corpus_texts]
	return BM25Okapi(tokenized_corpus)


def main():
	print("=== ĐANG CHẠY THỰC NGHIỆM BM25 TRÊN CRANFIELD ===")
	os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	if device == "cuda":
		print(f"[*] Thiết bị sử dụng: GPU - {torch.cuda.get_device_name(0)}")
	else:
		print("[!] CẢNH BÁO: Không nhận diện được GPU, BM25 sẽ chạy trên CPU.")

	print("\n[1/3] Đang tải dữ liệu Cranfield từ ir_datasets...")
	corpus, queries, qrels = load_cranfield()
	print(f"      -> Số lượng Document: {len(corpus)}")
	print(f"      -> Số lượng Query   : {len(queries)}")

	corpus_ids = list(corpus.keys())
	corpus_texts = [format_passage(corpus[doc_id]) for doc_id in corpus_ids]
	query_ids = list(queries.keys())

	print("[2/3] Đang xây dựng BM25 index...")
	if torch.cuda.is_available():
		torch.cuda.reset_peak_memory_stats()
		torch.cuda.synchronize()
	start_time = time.perf_counter()
	bm25 = build_bm25_index(corpus_texts)

	print("[3/3] Đang truy hồi và chấm điểm...")
	bm25_results = {}
	for q_id, q_text in queries.items():
		tokenized_query = tokenize(q_text)
		doc_scores = bm25.get_scores(tokenized_query)
		top_indices = np.argsort(doc_scores)[::-1][:TOP_RETRIEVE]
		bm25_results[q_id] = {
			corpus_ids[idx]: float(doc_scores[idx])
			for idx in top_indices
			if doc_scores[idx] > 0
		}

	if torch.cuda.is_available():
		torch.cuda.synchronize()
	mem_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0

	final_results = {
		MODEL_NAME: evaluate_results(bm25_results, qrels, MODEL_NAME)
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

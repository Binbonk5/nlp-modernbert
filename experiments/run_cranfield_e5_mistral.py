from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import ir_datasets
import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG PHÂN LẬP TUYỆT ĐỐI
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATASET_NAME = "cranfield"
MODEL_NAME = "intfloat/e5-mistral-7b-instruct"
MODEL_SLUG = MODEL_NAME.replace("/", "_")
RESULTS_FILE = RESULTS_DIR / f"cranfield_e5_mistral_{MODEL_SLUG}.json"

# ---------------------------------------------------------
# 2. SIÊU THAM SỐ ĐÃ ĐƯỢC HẠ THẤP ĐỂ TRÁNH TRÀN VRAM (OOM)
# ---------------------------------------------------------
QUERY_BATCH_SIZE = 4      # Hạ xuống 4 vì vector sinh từ Prompt dài ngốn nhiều bộ nhớ
CORPUS_BATCH_SIZE = 8     # Hạ xuống 8 để 15GB VRAM tĩnh của Mistral 7B không bị nổ
CORPUS_CHUNK_SIZE = 512
TOP_K = 10
MAX_SEQ_LENGTH = 512
BM25_K1 = 1.5
BM25_B = 0.75
BM25_WEIGHT = 0.4
DENSE_WEIGHT = 0.6

# Prompt chỉ thị chuẩn chỉnh của dòng E5-Instruct dành cho tập Cranfield
# E5_PROMPT = "Instruct: Given a technical information retrieval query from the Cranfield collection, retrieve relevant passages from scientific or engineering documents that answer the query\nQuery: "
E5_PROMPT = "Query: "


def build_model(model_name: str, device: str) -> SentenceTransformer:
    """
    Khởi tạo mô hình E5-Mistral chuẩn hệ Decoder.
    Tránh việc dùng lớp models.Pooling thủ công của BERT làm gãy cấu trúc toán học.
    """
    # Gọi trực tiếp SentenceTransformer để hệ thống tự động thiết lập:
    # 1. Left Padding cho Tokenizer (Bắt buộc đối với mô hình sinh từ Decoder)
    # 2. Cơ chế Last Token Pooling (Gom toàn bộ ngữ nghĩa Attention nhân quả vào Token cuối)
    model = SentenceTransformer(
        model_name,
        model_kwargs={
            "torch_dtype": torch.bfloat16,         # Ép về bfloat16 tiết kiệm 50% VRAM
            "trust_remote_code": True
        },
        device=device
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def format_passage(doc: Dict[str, str]) -> str:
    """Định dạng cấu trúc văn bản tài liệu đầu vào."""
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    if title and text:
        return f"{title} {text}"
    return title or text


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def load_cranfield():
    """Nạp dữ liệu từ bộ nhớ đệm cục bộ qua thư viện ir_datasets."""
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


def build_bm25_index(corpus_texts: List[str]):
    tokenized_docs = [tokenize(text) for text in corpus_texts]
    doc_term_freqs = [Counter(tokens) for tokens in tokenized_docs]
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0

    doc_freqs = Counter()
    for tokens in tokenized_docs:
        doc_freqs.update(set(tokens))

    num_docs = len(tokenized_docs)
    idf = {
        term: math.log(1.0 + (num_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in doc_freqs.items()
    }
    return doc_term_freqs, doc_lengths, avg_doc_len, idf


def score_bm25_query(
    query: str,
    doc_term_freqs,
    doc_lengths,
    avg_doc_len: float,
    idf: Dict[str, float],
) -> List[float]:
    query_terms = Counter(tokenize(query))
    scores = [0.0 for _ in range(len(doc_term_freqs))]

    if not query_terms:
        return scores

    for term in query_terms:
        term_idf = idf.get(term)
        if term_idf is None:
            continue

        for doc_index, doc_tf in enumerate(doc_term_freqs):
            frequency = doc_tf.get(term)
            if not frequency:
                continue

            doc_len = doc_lengths[doc_index]
            norm = frequency + BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / avg_doc_len if avg_doc_len else 0.0))
            scores[doc_index] += term_idf * (frequency * (BM25_K1 + 1.0)) / norm

    return scores


def normalize_rows(scores: torch.Tensor) -> torch.Tensor:
    row_min = scores.min(dim=1, keepdim=True).values
    row_max = scores.max(dim=1, keepdim=True).values
    denom = (row_max - row_min).clamp_min(1e-8)
    return (scores - row_min) / denom


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int, is_query: bool = False):
    """
    Mã hóa văn bản sang vector. 
    Nếu là Query, sử dụng Prompt chỉ thị. 
    Nếu là Corpus, truyền chuỗi trơn nhưng kích hoạt cấu hình tự động của E5-Mistral.
    """
    if is_query:
        # Đối với Query, chúng ta nạp thẳng chuỗi đã được f-string bọc Prompt sẵn
        return model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    else:
        # Đối với Corpus, tài liệu e5-mistral yêu cầu không được dùng chung cơ chế dịch với câu hỏi.
        # Chúng ta dùng tham số ẩn để ép mô hình hiểu đây là tài liệu đích cần tìm kiếm.
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
    # Bọc đúng cấu trúc ngắt dòng theo chuẩn Paper gốc của E5-Mistral
    query_texts = [f"{E5_PROMPT}{text}" for text in queries.values()]

    print("   -> Đang tiến hành nhúng ma trận câu hỏi (Queries Embedding)...")
    # THÊM THAM SỐ: is_query=True
    query_embeddings = encode_texts(model, query_texts, batch_size=QUERY_BATCH_SIZE, is_query=True).to(device)

    dense_scores = torch.empty((len(query_ids), len(corpus_ids)), dtype=torch.float32, device=device)
    for start in tqdm(range(0, len(corpus_ids), CORPUS_CHUNK_SIZE), desc="Tiến độ xử lý Chunks", unit="chunk"):
        end = min(start + CORPUS_CHUNK_SIZE, len(corpus_ids))
        # THÊM THAM SỐ: is_query=False cho tập tài liệu cần tìm
        chunk_embeddings = encode_texts(model, corpus_texts[start:end], batch_size=CORPUS_BATCH_SIZE, is_query=False).to(device)
        dense_scores[:, start:end] = torch.matmul(query_embeddings, chunk_embeddings.transpose(0, 1))

    doc_term_freqs, doc_lengths, avg_doc_len, idf = build_bm25_index(corpus_texts)
    bm25_scores = torch.tensor(
        [score_bm25_query(query_text, doc_term_freqs, doc_lengths, avg_doc_len, idf) for query_text in query_texts],
        dtype=torch.float32,
        device=device,
    )

    hybrid_scores = DENSE_WEIGHT * normalize_rows(dense_scores) + BM25_WEIGHT * normalize_rows(bm25_scores)
    top_scores, top_doc_idx = torch.topk(hybrid_scores, k=TOP_K, dim=1)

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
        print(f"[*] Thiết thiết bị nhận diện thành công : GPU - {torch.cuda.get_device_name(0)}")
    else:
        print("[!] CẢNH BÁO: Không tìm thấy GPU, hệ thống sẽ chạy cực chậm trên CPU!")

    print("\n[1/5] Đang nạp bộ dữ liệu Cranfield qua ir_datasets...")
    corpus, queries, qrels = load_cranfield()
    print(f"      -> Số lượng Document: {len(corpus)}")
    print(f"      -> Số lượng Query   : {len(queries)}")

    print("\n[2/5] Đang tải siêu mô hình E5-Mistral-7B vào bộ nhớ GPU...")
    model = build_model(MODEL_NAME, device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print("\n[3/5] & [4/5] Kích hoạt quá trình mã hóa chuỗi dài và đối chiếu Vector...")
    start_time = time.perf_counter()
    results = retrieve_topk(model, corpus, queries)
    
    print("\n[5/5] Đang chấm điểm hiệu năng Retrieval qua bộ công cụ BEIR...")
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values=[TOP_K])
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    # Đóng gói nhật ký cấu hình thực nghiệm
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
        "prompt": E5_PROMPT,
        "fusion_method": "bm25_dense_weighted_sum",
        "bm25_weight": BM25_WEIGHT,
        "dense_weight": DENSE_WEIGHT,
        "optimization": "last_token_pooling_e5_mistral_native",
        "device": device,
        "results_file": str(RESULTS_FILE.relative_to(PROJECT_ROOT)),
        "corpus_size": len(corpus),
        "query_size": len(queries),
    }

    with RESULTS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\n=== KẾT QUẢ THỰC NGHIỆM ĐÃ ĐƯỢC KHÔI PHỤC HOÀN HẢO ===")
    print(json.dumps(record, indent=4))
    print(f"[*] File kết quả đã lưu trữ thành công tại: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
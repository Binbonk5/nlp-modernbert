from __future__ import annotations

import json
import os
import time
from pathlib import Path

import evaluate
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "sst2_bert_large"
TEST_CSV_PATH = DATA_DIR / "glue-sst2" / "test.csv"

MODEL_NAME = "google-bert/bert-large-uncased"
MODEL_SLUG = MODEL_NAME.replace("/", "_")
RESULTS_FILE = RESULTS_DIR / f"sst2_bert_large_{MODEL_SLUG}.json"
HF_DATASET_CACHE_DIR = PROJECT_ROOT / "checkpoints" / "hf_datasets"
DATASET_NAME = "sst2"
BATCH_SIZE = 16
MAX_LENGTH = 256
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42


def prepare_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.cls_token
    return tokenizer


def tokenize_split(split, tokenizer):
    def _tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=MAX_LENGTH)

    tokenized = split.map(_tokenize, batched=True, remove_columns=[column for column in split.column_names if column in {"sentence", "idx"}])
    if "label" in tokenized.column_names:
        tokenized = tokenized.rename_column("label", "labels")
    return tokenized


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = torch.from_numpy(logits).argmax(dim=-1)
    return metric.compute(predictions=predictions.numpy(), references=labels)


def main():
    print(f"=== BẮT ĐẦU CHẠY THỰC NGHIỆM: {MODEL_NAME} ===")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[*] Thiết bị sử dụng : GPU - {torch.cuda.get_device_name(0)}")
    else:
        print("[!] CẢNH BÁO: Không nhận diện được GPU, đang chạy bằng CPU!")

    print("\n[1/5] Đang tải tokenizer và mô hình...")
    tokenizer = prepare_tokenizer(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id

    print("[2/5] Đang tải dữ liệu SST-2...")
    raw_dataset = load_dataset("nyu-mll/glue", DATASET_NAME, cache_dir=str(HF_DATASET_CACHE_DIR))
    train_dataset = tokenize_split(raw_dataset["train"], tokenizer)
    eval_dataset = tokenize_split(raw_dataset["validation"], tokenizer)
    print(f"      -> Số lượng train    : {len(train_dataset)}")
    print(f"      -> Số lượng validation: {len(eval_dataset)}")
    print("      -> Đang nạp thêm tập test.csv để kiểm tra cuối cùng...")
    raw_test_dataset = load_dataset("csv", data_files={"test": str(TEST_CSV_PATH)}, cache_dir=str(HF_DATASET_CACHE_DIR))
    test_dataset = tokenize_split(raw_test_dataset["test"], tokenizer)
    print(f"      -> Số lượng test      : {len(test_dataset)}")

    print("[3/5] Đang khởi tạo trainer và cấu hình huấn luyện...")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        bf16=torch.cuda.is_available(),
        fp16=False,
        tf32=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=[],
        dataloader_num_workers=4,
        seed=SEED,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator, compute_metrics=compute_metrics)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print("[4/5] Bắt đầu huấn luyện...")
    start_time = time.perf_counter()
    trainer.train()
    print("[5/5] Đang đánh giá mô hình...")
    metrics = trainer.evaluate()
    test_metrics = trainer.predict(test_dataset).metrics
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    trainer.save_model(str(CHECKPOINT_DIR))
    tokenizer.save_pretrained(str(CHECKPOINT_DIR))

    record = {
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "metric_name": "accuracy",
        "score": round(float(metrics.get("eval_accuracy", 0.0)), 6),
        "test_metric_name": "accuracy",
        "test_score": round(float(test_metrics.get("test_accuracy", 0.0)), 6),
        "time_sec": round(time.perf_counter() - start_time, 6),
        "mem_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "checkpoint_dir": str(CHECKPOINT_DIR.relative_to(PROJECT_ROOT)),
        "results_file": str(RESULTS_FILE.relative_to(PROJECT_ROOT)),
        "test_file": str(TEST_CSV_PATH.relative_to(PROJECT_ROOT)),
        "device": device,
    }

    with RESULTS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\n=== KẾT QUẢ ===")
    print(json.dumps(record, indent=4))
    print(f"[*] Đã lưu kết quả tại: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
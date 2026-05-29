from __future__ import annotations

import json
import os
import time
from pathlib import Path

import evaluate
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

# ---------------------------------------------------------
# 1. CẤU HÌNH CỨNG PHÂN LẬP TUYỆT ĐỐI (LARGE CHUẨN PAPER)
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "sst2_modernbert_large"
TEST_CSV_PATH = DATA_DIR / "glue-sst2" / "test.csv"

MODEL_NAME = "AnswerDotAI/ModernBERT-large"  # Mô hình Large
MODEL_SLUG = MODEL_NAME.replace("/", "_")
RESULTS_FILE = RESULTS_DIR / f"sst2_modernbert_large_{MODEL_SLUG}.json"
HF_DATASET_CACHE_DIR = PROJECT_ROOT / "checkpoints" / "hf_datasets"

DATASET_NAME = "sst2"
BATCH_SIZE = 32         # Đã nâng lên 32: GPU A100 dư sức nuốt gọn bản Large ở batch size này
MAX_LENGTH = 256
NUM_EPOCHS = 3
LEARNING_RATE = 1e-5    # ĐÃ SỬA: Mức LR 5e-5 tối ưu nhất cho cấu trúc mã hóa dòng Large
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
        print(f"[*] Thiết thiết bị sử dụng : GPU - {torch.cuda.get_device_name(0)}")

    print("\n[1/5] Đang khởi tạo Tokenizer...")
    tokenizer = prepare_tokenizer(MODEL_NAME)

    # ---------------------------------------------------------
    # CƠ CHẾ KIỂM TRA CHECKPOINT THÔNG MINH
    # ---------------------------------------------------------
    is_trained = (CHECKPOINT_DIR / "config.json").exists()
    if is_trained:
        print(f"[*] Tìm thấy mô hình đã được Fine-tune sẵn tại: {CHECKPOINT_DIR}")
        print("      -> Đang nạp mô hình từ checkpoint (Bỏ qua bước Huấn luyện)...")
        model_load_path = str(CHECKPOINT_DIR)
    else:
        print(f"[*] Chưa có checkpoint cũ. Sẽ tiến hành huấn luyện từ mô hình thô: {MODEL_NAME}")
        model_load_path = MODEL_NAME

    model = AutoModelForSequenceClassification.from_pretrained(
        model_load_path, 
        num_labels=2, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, 
        trust_remote_code=True, 
        attn_implementation="flash_attention_2"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    print("\n[2/5] Đang tải dữ liệu SST-2 từ bộ nhớ đệm...")
    raw_dataset = load_dataset("nyu-mll/glue", DATASET_NAME, cache_dir=str(HF_DATASET_CACHE_DIR))
    train_dataset = tokenize_split(raw_dataset["train"], tokenizer)
    eval_dataset = tokenize_split(raw_dataset["validation"], tokenizer)
    print(f"      -> Số lượng train     : {len(train_dataset)}")
    print(f"      -> Số lượng validation: {len(eval_dataset)}")

    print("      -> Đang nạp thêm tập test.csv để kiểm tra cuối cùng...")
    raw_test_dataset = load_dataset("csv", data_files={"test": str(TEST_CSV_PATH)}, cache_dir=str(HF_DATASET_CACHE_DIR))
    test_dataset = tokenize_split(raw_test_dataset["test"], tokenizer)
    print(f"      -> Số lượng test      : {len(test_dataset)}")

    print("\n[3/5] Đang thiết lập cấu hình Trainer tối ưu tuyệt đối cho dòng Large...")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ĐÃ SỬA: Cập nhật đồng bộ bộ tham số vàng, bổ sung weight_decay và cấu hình tối ưu DDP
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        
        # ĐỒNG BỘ TOÀN DIỆN CHẾ ĐỘ CHÍNH XÁC CAO NHƯ NOTEBOOK
        bf16=torch.cuda.is_available(),
        bf16_full_eval=torch.cuda.is_available(),
        fp16=False,
        tf32=torch.cuda.is_available(),
        gradient_checkpointing=True,     # ĐÃ BẬT: Bắt buộc bật để bảo vệ VRAM khi chạy bản Large nặng
        dataloader_num_workers=4,
        
        # BỘ SIÊU THAM SỐ ĐỘC QUYỀN CHỐNG SỤP ĐỔ GRADIENT
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        lr_scheduler_type="linear",
        weight_decay=1e-6,               # ĐÃ BỔ SUNG: Kiểm soát biên độ tạ, ngăn Overfitting
        
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=[],
        seed=SEED,
    )

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        processing_class=tokenizer,     # Cập nhật chuẩn Transformers mới
        data_collator=data_collator, 
        compute_metrics=compute_metrics
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    # ---------------------------------------------------------
    # LUỒNG ĐIỀU HƯỚNG TẬP LỆNH CHẠY
    # ---------------------------------------------------------
    if not is_trained:
        print("\n[4/5] Bắt đầu quá trình Huấn luyện (Fine-tuning ModernBERT-Large)...")
        trainer.train()
        print("[*] Huấn luyện xong! Đang đóng gói lưu mô hình...")
        trainer.save_model(str(CHECKPOINT_DIR))
        tokenizer.save_pretrained(str(CHECKPOINT_DIR))
    else:
        print("\n[4/5] Bỏ qua bước huấn luyện theo cấu hình hệ thống.")

    print("\n[5/5] Bắt đầu bước Đánh giá hiệu năng (Inference Eval)...")
    metrics = trainer.evaluate()
    test_metrics = trainer.predict(test_dataset).metrics
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time

    record = {
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "metric_name": "accuracy",
        "score": round(float(metrics.get("eval_accuracy", 0.0)), 6),
        "test_metric_name": "accuracy",
        "test_score": round(float(test_metrics.get("test_accuracy", 0.0)), 6),
        "time_sec": round(elapsed_time, 6),
        "mem_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "num_epochs": NUM_EPOCHS if not is_trained else 0,
        "learning_rate": LEARNING_RATE,
        "checkpoint_dir": str(CHECKPOINT_DIR.relative_to(PROJECT_ROOT)),
        "results_file": str(RESULTS_FILE.relative_to(PROJECT_ROOT)),
        "test_file": str(TEST_CSV_PATH.relative_to(PROJECT_ROOT)),
        "device": device,
        "optimization": "flash_attention_2_large_optimized_config",
    }

    with RESULTS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\n=== KẾT QUẢ ĐÃ ĐƯỢC CẬP NHẬT CHUẨN PAPER (LARGE) ===")
    print(json.dumps(record, indent=4))


if __name__ == "__main__":
    main()
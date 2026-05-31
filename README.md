# Reproduce Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference

Môn NLP - Nhóm 5

- Cáp Kim Hải Anh - 23520036
- Hoàng Đức Dũng - 23520328
- Nguyễn Thái Sơn - 23521356
- Bùi Ngọc Thiên Thanh - 23521436

Paper: https://arxiv.org/pdf/2412.13663

## GPU Requirement

Yêu cầu tài nguyên GPU: GTX 4090 24GB trở lên.

## Repository Tree

```text
nlp-modernbert/
├── data/
├── experiments/
├── results/
└── scripts/
```

## Setup

```bash
git clone https://github.com/Binbonk5/nlp-modernbert.git
cd nlp-modernbert
conda create -n nlp-modernbert python=3.10 -y
conda activate nlp-modernbert
pip install -U pip
pip install -r requirements.txt
pip install --no-build-isolation flash-attn>=2.6.3
```

## Run

```bash
# SST-2
python experiments/run_sst2_bert_base.py
python experiments/run_sst2_bert_large.py
python experiments/run_sst2_modernbert_base.py
python experiments/run_sst2_modernbert_large.py

# TREC-COVID
python experiments/run_trec_covid_bert_base.py
python experiments/run_trec_covid_bert_large.py
python experiments/run_trec_covid_modernbert_base.py
python experiments/run_trec_covid_modernbert_large.py

# Cranfield
python experiments/run_cranfield_bge_m3_large.py
python experiments/run_cranfield_e5_mistral.py
python experiments/run_cranfield_modernbert_large.py

# Aggregate results
python scripts/aggregate.py --results-dir results --output results_report.csv
```

The SST-2 evaluation script reads the local test split from `data/glue-sst2/test.csv`.

If you want to launch multi-GPU training, use `accelerate launch` with the relevant experiment file, for example:

```bash
accelerate launch --num_processes 4 experiments/run_sst2_modernbert_base.py
```

## References

1. Reproduce Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference. https://arxiv.org/pdf/2412.13663
2. This repository: https://github.com/Binbonk5/nlp-modernbert

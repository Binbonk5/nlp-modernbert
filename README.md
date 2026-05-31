# REPRODUCE *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference*
[![arXiv](https://img.shields.io/badge/arXiv-2412.13663-B31B1B.svg)](https://arxiv.org/abs/2412.13663)
[![ACL Anthology](https://img.shields.io/badge/ACL--Anthology-2025.acl--long.127-115926.svg)](https://aclanthology.org/2025.acl-long.127/)

Môn NLP - Nhóm 5

- Hoàng Đức Dũng - 23520328
- Cáp Kim Hải Anh - 23520036
- Nguyễn Thái Sơn - 23521356
- Bùi Ngọc Thiên Thanh - 23521436

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

## Getting Started

### 1. Clone nlp-bert

```bash
git clone https://github.com/Binbonk5/nlp-modernbert.git
cd nlp-modernbert
```

### 2. Install Dependencies

```bash
conda create -n nlp-modernbert python=3.10 -y
conda activate nlp-modernbert
pip install -U pip
pip install -r requirements.txt
pip install --no-build-isolation flash-attn>=2.6.3
```

### 3. Data: 
- Auto download when you run 
- The SST-2 evaluation script reads the local test split from `data/glue-sst2/test.csv`.

### 4. Evaluation

#### SST-2

```bash
python experiments/run_sst2_bert_base.py
python experiments/run_sst2_bert_large.py
python experiments/run_sst2_modernbert_base.py
python experiments/run_sst2_modernbert_large.py
```

#### TREC-COVID

```bash
python experiments/run_trec_covid_bert_base.py
python experiments/run_trec_covid_bert_large.py
python experiments/run_trec_covid_modernbert_base.py
python experiments/run_trec_covid_modernbert_large.py
```

#### Cranfield

```bash
python experiments/run_cranfield_bge_m3_large.py
python experiments/run_cranfield_e5_mistral.py
python experiments/run_cranfield_modernbert_large.py
```

#### Aggregate Results

```bash
python scripts/aggregate.py --results-dir results --output results_report.csv
```

If you want to launch multi-GPU training, use `accelerate launch` with the relevant experiment file, for example:

```bash
accelerate launch --num_processes 4 experiments/run_sst2_modernbert_base.py
```

## References

1. Reproduce Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference. https://arxiv.org/pdf/2412.13663
2. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. https://arxiv.org/abs/1810.04805
3. BGE-M3: A Unified Multilingual Multi-Function Dense Retrieval Model. https://arxiv.org/abs/2402.03216
4. E5-Mistral: A Strong Text Embedding Model. https://huggingface.co/intfloat/e5-mistral-7b-instruct
5. This repository: https://github.com/Binbonk5/nlp-modernbert

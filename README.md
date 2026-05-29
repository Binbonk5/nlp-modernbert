# nlp-modernbert

## Layout

- `experiments/`: one standalone Python file per model + dataset combination
- `results/`: one JSON result file per experiment
- `checkpoints/`: saved fine-tuned models and Hugging Face caches
- `data/`: local datasets used by the SST-2 and retrieval experiments
- `scripts/aggregate.py`: combines JSON outputs into a CSV report

## Setup

```bash
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

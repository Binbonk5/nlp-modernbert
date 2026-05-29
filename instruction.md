# NLP ModernBERT

Current workspace convention:

```text
nlp-modernbert/
├── checkpoints/
├── data/
├── experiments/
├── results/
├── scripts/
│   └── aggregate.py
└── requirements.txt
```

Rules:

- Each model + dataset pair lives in its own standalone Python script under `experiments/`.
- Scripts do not use `argparse`; all model, dataset, batch size, prompt, and optimization settings are hardcoded.
- Each script writes exactly one JSON result file under `results/`.
- Fine-tuned runs save artifacts under `checkpoints/`.
- `scripts/aggregate.py` aggregates the JSON result files into a CSV report.

Current experiment files:

- `experiments/run_sst2_bert_base.py`
- `experiments/run_sst2_bert_large.py`
- `experiments/run_sst2_modernbert_base.py`
- `experiments/run_sst2_modernbert_large.py`
- `experiments/run_trec_covid_bert_base.py`
- `experiments/run_trec_covid_bert_large.py`
- `experiments/run_trec_covid_modernbert_base.py`
- `experiments/run_trec_covid_modernbert_large.py`
- `experiments/run_msmacro_bge_m3_large.py`
- `experiments/run_msmarco_modernbert_large.py`
- `experiments/run_msmarco_e5_mistral.py`
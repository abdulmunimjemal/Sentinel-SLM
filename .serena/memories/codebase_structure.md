# Codebase Structure

- `src/sentinel/` core library
  - `data/download.py`: dataset fetching from HuggingFace
  - `data/processing.py`: map raw data to 8-category taxonomy
  - `data/synthetic.py`: synthetic samples via OpenRouter
  - `train/train_rail_a.py`: Rail A training logic
  - `utils/taxonomy.py`: category enums (source of truth)
- `scripts/` workflow scripts
  - `run_pipeline.py`: download/process pipeline
  - `prepare_rail_a_data.py`: extract Rail A data from main dataset
  - `prepare_rail_a_external.py`: download external jailbreak datasets
  - `enrich_language.py`: add language detection
  - `finalize_rail_a.py`: organize model checkpoints
  - `inspect_rail_a.py`: dataset stats
  - `check_model.py`: model architecture checks
- `notebooks/` exploration/training/eval notebooks
  - `04_train_rail_a.ipynb`, `05_test_rail_a.ipynb` are key Rail A training/eval notebooks
- `docs/` architecture, taxonomy, training results, implementation plan
- `data/`, `models/`, `assets/` store datasets, trained artifacts, media

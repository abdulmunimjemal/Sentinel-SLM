# Suggested Commands

Environment
- Create venv: `python3 -m venv .venv`
- Activate: `source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Prefer running with: `.venv/bin/python ...`

Main data pipeline
- Download: `.venv/bin/python scripts/run_pipeline.py --download`
- Process: `.venv/bin/python scripts/run_pipeline.py --process`
- All: `.venv/bin/python scripts/run_pipeline.py --all`

Rail A data prep
- External jailbreak datasets: `.venv/bin/python scripts/prepare_rail_a_external.py`
- Extract from main dataset: `.venv/bin/python scripts/prepare_rail_a_data.py`
- Language enrichment: `.venv/bin/python scripts/enrich_language.py`
- Inspect dataset: `.venv/bin/python scripts/inspect_rail_a.py`

Model finalization
- Finalize checkpoint: `.venv/bin/python scripts/finalize_rail_a.py --checkpoint models/rail_a_v3/checkpoint-XXX`

Evaluation
- Run notebook: `jupyter notebook notebooks/05_test_rail_a.ipynb`

Notes
- No explicit lint/format/test CLI commands found beyond notebooks.

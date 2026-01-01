# Sentinel-SLM: Agent Instructions

**Welcome, Agent.**
This file is your primary orientation guide. It describes the project structure, development workflow, and current state. 
**READ THIS FIRST before making changes.**

---

## 1. Project Context
**Goal**: Build "Sentinel-SLM", a multilingual content moderation model (1.2B - 3B param range) trained on 8 safety categories.
**Status**: **Phase 1 (Data Prep) Complete**. We have a `data/processed/final_augmented_dataset.parquet` with >1.6M samples.
**Next Phase**: Phase 2 (Training/Fine-tuning).

## 2. Environment
*   **Python**: 3.9+
*   **Virtual Env**: `.venv` (Always use this: `.venv/bin/python`)
*   **Dependencies**: Listed in `requirements.txt`.
*   **Environment Variables**: Managed in `.env`.
    *   `HF_TOKEN`: Required for downloading gated datasets (BeaverTails, KoalaAI).
    *   `OPENROUTER_API_KEY`: Required only for synthetic data generation.

## 3. Codebase Structure (`src/sentinel`)
We use a modular package structure. Do not create loose scripts in root.

*   `src/sentinel/data/download.py`: Code to fetch data from HuggingFace.
*   `src/sentinel/data/processing.py`: Logic to map raw columns -> 8-category taxonomy. Multi-step pipeline.
*   `src/sentinel/data/synthetic.py`: Logic for generating synthetic samples using OpenRouter (LLama 3, etc).
*   `src/sentinel/utils/taxonomy.py`: **Source of Truth** for Category Enums (0-8) and mapping rules.

## 4. Key Commands (The "One Script")
Use `scripts/run_pipeline.py` for all data operations. **Do not run `src` modules directly.**

```bash
# 1. Download Datasets (Public & Gated)
.venv/bin/python scripts/run_pipeline.py --download

# 2. Process & Map to Taxonomy  (Creates unified_dataset.parquet)
.venv/bin/python scripts/run_pipeline.py --process

# 3. Generate Synthetic Data (Optional/Gap Filling)
.venv/bin/python scripts/run_pipeline.py --synthetic --count 50

# 4. Run Everything
.venv/bin/python scripts/run_pipeline.py --all
```

## 5. Dataset Information
*   **Final Output**: `data/processed/final_augmented_dataset.parquet`
*   **Schema**:
    *   `text` (str): The input text/prompt.
    *   `labels` (List[int]): List of category IDs (e.g., `[1, 5]`).
    *   `source` (str): Origin dataset.

## 6. Best Practices
1.  **Type Hinting**: Use `typing` (List, Dict, Optional) in all new functions.
2.  **Logging**: Use `logging` instead of `print`.
3.  **Pathing**: Always use `os.path.join` and relative paths from `PROJECT_ROOT`.
4.  **Updates**: When adding a new dataset, update `download.py` AND `taxonomy.py`, then re-run processing.

## 7. Current Task List
Refer to `task.md` (in brain Artifacts) for the active checklist.

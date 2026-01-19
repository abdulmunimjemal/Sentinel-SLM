# Sentinel-SLM: Agent Instructions

**Welcome, Agent.**
This file is your primary orientation guide. It describes the project structure, development workflow, and current state. 
**READ THIS FIRST before making changes.**

---

## 1. Project Context
**Goal**: Build "Sentinel-SLM", a multilingual content moderation model using a dual-rail architecture:
- **Rail A (Input Guard)**: Binary classifier for prompt injection/jailbreak detection
- **Rail B (Policy Guard)**: Multi-label classifier for 8 safety categories

**Status**: 
- **Phase 1 (Data Prep)**: Complete. Main dataset: `data/processed/final_augmented_dataset.parquet` (>1.6M samples)
- **Phase 2 (Rail A Training)**: Complete. Model: `models/rail_a_v3/final/`
- **Phase 3 (Rail B Training)**: Pending

## 2. Environment
*   **Python**: 3.9+
*   **Virtual Env**: `.venv` (Always use this: `.venv/bin/python`)
*   **Dependencies**: Listed in `requirements.txt`.
*   **Environment Variables**: Managed in `.env`.
    *   `HF_TOKEN`: Required for downloading gated datasets (BeaverTails, KoalaAI).
    *   `OPENROUTER_API_KEY`: Required only for synthetic data generation.

## 3. Codebase Structure

### Source Code (`src/sentinel/`)
*   `data/download.py`: Fetch datasets from HuggingFace
*   `data/processing.py`: Map raw data to 8-category taxonomy
*   `data/synthetic.py`: Generate synthetic samples via OpenRouter
*   `train/train_rail_a.py`: Rail A training logic with custom model wrapper
*   `utils/taxonomy.py`: **Source of Truth** for Category Enums (0-8)

### Scripts (`scripts/`)
*   `run_pipeline.py`: Main pipeline for data download/processing
*   `prepare_rail_a_data.py`: Extract Rail A data from main dataset
*   `prepare_rail_a_external.py`: Download external jailbreak datasets (5 sources)
*   `enrich_language.py`: Add language detection via FastText
*   `finalize_rail_a.py`: Organize model checkpoints for deployment
*   `inspect_rail_a.py`: Analyze Rail A dataset statistics
*   `check_model.py`: Verify model architecture

### Notebooks (`notebooks/`)
*   `01_data_preparation.ipynb`: Initial data exploration
*   `02_synthetic_inspection.ipynb`: Review synthetic data quality
*   `03_eda.ipynb`: Exploratory data analysis
*   `04_train_rail_a.ipynb`: **Rail A training notebook**
*   `05_test_rail_a.ipynb`: **Rail A evaluation with metrics**

## 4. Rail A Data Pipeline

Rail A uses data from **5 external jailbreak datasets**:
1. `deepset/prompt-injections` - Labeled injection dataset
2. `JailbreakBench/JBB-Behaviors` - Jailbreak behavior goals
3. `TrustAIRLab/in-the-wild-jailbreak-prompts` - Real-world prompts
4. `Simsonsun/JailbreakPrompts` - Curated jailbreak prompts
5. `yanismiraoui/prompt_injections` - Additional samples

**Safe samples** for balancing come from:
- `databricks/databricks-dolly-15k`
- `tatsu-lab/alpaca`

```bash
# Option A: Download external jailbreak datasets
.venv/bin/python scripts/prepare_rail_a_external.py

# Option B: Extract from main dataset (if attacks exist there)
.venv/bin/python scripts/prepare_rail_a_data.py
```

**Label Convention**: `0 = Safe`, `1 = Attack`

## 5. Rail A Model

*   **Base Model**: `LiquidAI/LFM2-350M`
*   **Architecture**: Base model + LoRA adapters + Classification head
*   **Output Directory**: `models/rail_a_v3/final/`
    *   `tokenizer_config.json`, `tokenizer.json`, etc.
    *   `adapter_model.safetensors`, `adapter_config.json` (LoRA)
    *   `classifier.pt` (Classification head)

## 6. Key Commands

```bash
# Main Data Pipeline
.venv/bin/python scripts/run_pipeline.py --download
.venv/bin/python scripts/run_pipeline.py --process
.venv/bin/python scripts/run_pipeline.py --all

# Rail A Data Preparation
.venv/bin/python scripts/prepare_rail_a_external.py
.venv/bin/python scripts/enrich_language.py

# Model Finalization
.venv/bin/python scripts/finalize_rail_a.py --checkpoint models/rail_a_v3/checkpoint-XXX
```

## 7. Dataset Information

### Main Dataset
*   **Path**: `data/processed/final_augmented_dataset.parquet`
*   **Schema**: `text`, `labels` (List[int]), `source`

### Rail A Dataset
*   **Path**: `data/processed/rail_a_external.parquet`
*   **Schema**: `text`, `target` (0 or 1), `source`
*   **Size**: ~8,000 samples (balanced)

## 8. Best Practices
1.  **Type Hinting**: Use `typing` module in all functions
2.  **Logging**: Use `logging` instead of `print`
3.  **Pathing**: Use `os.path.join` with relative paths
4.  **Testing**: Run `05_test_rail_a.ipynb` after training to verify metrics

## 9. Taxonomy (8 Categories)
```
0: SAFE
1: HATE_EXTREMISM
2: HARASSMENT
3: SEXUAL
4: CHILD_SAFETY
5: VIOLENCE
6: ILLEGAL
7: PRIVACY
8: PROMPT_ATTACK (Rail A target)
```

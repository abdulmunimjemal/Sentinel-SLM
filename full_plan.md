# Data Preparation Plan: Sentinel-SLM

## Goal
Prepare a high-quality, multilingual dataset for 8-category content moderation by **first aggregating existing public datasets** and only then filling gaps with synthetic data.

## 1. Environment & Setup
- **Virtual Env**: `.venv` (created)
- **Repo Structure**: `data/raw`, `data/processed`, `scripts/`
- **Dependencies**: `datasets`, `pandas`, `openai`

## 2. Priority 1: Fetching Public Data
We will download and standardize these reputable datasets first.

| Dataset | Priority | Target Categories |
| :--- | :--- | :--- |
| **BeaverTails** | High | Sexual, Child Safety, Illegal, Privacy, Violence |
| **Jigsaw Toxic Severity** | High | Hate, Harassment, Sexual, Violence |
| **JailbreakBench** | Medium | Prompt Attack (Category 8) |
| **XNLI / TweetEval** | Low | Multilingual Hate (if needed) |

**Script**: `scripts/download_data.py` (Ready to run)
- Action: Downloads datasets via Hugging Face `datasets` library.
- Storage: Saves as `.parquet` in `data/raw/`.

## 3. Priority 2: Data Mapping (Standardization)
Once raw data is downloaded, we must map it to our 8-category taxonomy.
**Script**: `scripts/map_labels.py` (To be created)
- **Input**: Raw parquet files.
- **Logic**: Use `src/utils/taxonomy.py` mapping rules.
- **Output**: `data/processed/unified_dataset.parquet`.

## 4. Priority 3: Gap Analysis & Synthetic Generation
Only *after* we have the unified dataset, we check class balances.
- **Gap analysis**: Count samples per Category + Language.
- **Synthesis**: Use OpenRouter to generate samples *only* for under-represented buckets (likely Privacy, Non-English categories).

## 5. Next Steps execution
1.  [x] Create Venv & Git
2.  [ ] Run `scripts/download_data.py`
3.  [ ] Inspect downloaded data sizes
4.  [ ] Create `scripts/map_labels.py`

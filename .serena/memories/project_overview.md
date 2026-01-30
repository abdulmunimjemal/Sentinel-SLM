# Sentinel-SLM Overview

- Purpose: Multilingual content moderation and guardrails for LLM deployments using a dual-rail architecture.
- Rail A (Input Guard): Binary classifier for prompt injection/jailbreak detection.
- Rail B (Policy Guard): Multi-label classifier for 8 safety categories (pending training).
- Status: Phase 1 data prep complete; Phase 2 Rail A training complete; Phase 3 Rail B training pending.
- Data: Main dataset `data/processed/final_augmented_dataset.parquet` (>1.6M samples). Rail A dataset `data/processed/rail_a_external.parquet` (~8k balanced samples).
- Model: Base `LiquidAI/LFM2-350M` with LoRA adapters + classification head; Rail A final model in `models/rail_a_v3/final/`.
- Tech stack: Python 3.9+, datasets/pandas/pyarrow, scikit-learn, peft/accelerate/evaluate, matplotlib/seaborn, langdetect, notebook/ipykernel, python-dotenv, openai/distilabel.

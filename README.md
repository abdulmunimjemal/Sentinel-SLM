# Sentinel-SLM
> **Multilingual Text Content Moderation with Small Language Models**

## Overview
Sentinel-SLM is a system designed to detect harmful content across 8 policy categories (Hate, Harassment, Sexual, Child Safety, etc.) using efficient Small Language Models (Liquid LFM2 / Phi-3) optimized for edge deployment.

## Project Structure
```
.
├── src/sentinel/         # Main package
│   ├── data/             # Data ingestion, processing, synthetic generation
│   └── utils/            # Taxonomy and helpers
├── scripts/              # CLI entry points (run_pipeline.py)
├── notebooks/            # Jupyter notebooks for analysis
├── data/                 # Dataset storage (Raw & Processed)
└── tests/                # (Planned) Unit tests
```

## Setup
1. **Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **API Key**:
   Create a `.env` file with your OpenRouter key:
   ```
   OPENROUTER_API_KEY=sk-...
   ```

## Usage
Run the unified data pipeline:
```bash
# Download and Process everything
python scripts/run_pipeline.py --all

# Generate Synthetic Data (filling gaps)
python scripts/run_pipeline.py --synthetic --count 50
```

## Taxonomy
The system classifies text into:
1. Hate & Extremism
2. Harassment & Bullying
3. Sexual Content
4. Child Safety & Exploitation
5. Violence & Gore
6. Illegal Activities
7. Privacy Violations
8. Prompt Attacks (Jailbreaks)

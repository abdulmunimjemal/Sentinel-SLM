# Sentinel-SLM
> **Multilingual Text Content Moderation with Small Language Models**

## Overview
Sentinel-SLM is a system designed to detect harmful content across 8 policy categories (Hate, Harassment, Sexual, Child Safety, etc.) using efficient Small Language Models (Liquid LFM2 / Phi-3) optimized for edge deployment.

## Documentation
- 📂 [**Project Plan**](docs/implementation_plan.md) - Full research report and implementation roadmap.
- 📚 [**Data Taxonomy**](docs/taxonomy.md) - Definitions of the 8 policy categories and dataset schema.
- 🤖 [**Agents Guide**](AGENTS.md) - Instructions for AI agents working on this repo.

## Project Structure
```
.
├── src/sentinel/         # Main package (data, utils)
├── scripts/              # CLI entry point (run_pipeline.py)
├── notebooks/            # Jupyter notebooks for analysis
├── docs/                 # Documentation
└── data/                 # Dataset storage (Raw & Processed)
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

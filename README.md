<div align="center">
  <img src="assets/logo.png" alt="Sentinel-SLM Logo" width="200"/>
  <h1>Sentinel-SLM: Multilingual Content Moderation & Guardrails</h1>
</div>

> **Safety for the Open Web.**
> A robust, open-source SLM (Small Language Model) specialized in multilingual toxicity detection, jailbreak categorization, and safety guardrails.

Sentinel-SLM is a specialized 8-category content moderation model designed to act as a lightweight, high-performance "guard" for Large Language Model (LLM) deployments. It implements a **Dual-Rail Architecture** to secure both user inputs (Prompt Injections) and model outputs (Policy Violations).

---

## 🛡️ Taxonomy (8 Categories)

Sentinel-SLM is fine-tuned to detect the following risk categories:

| ID | Category | Description |
| :--- | :--- | :--- |
| **0** | **Safe** | Benign content, no risk detected. |
| **1** | **Hate & Extremism** | Hate speech, racial slurs, extremist ideologies, discrimination. |
| **2** | **Harassment** | Bullying, severe toxicity, personal attacks, insults. |
| **3** | **Sexual Content** | sexually explicit material, NSFW content, pornography. |
| **4** | **Child Safety** | CSAM, child exploitation, grooming attempts. |
| **5** | **Violence** | Gore, brutal violence, physical threats, self-harm. |
| **6** | **Illegal Activities** | Drug trade, weapons manufacturing, financial crimes. |
| **7** | **Privacy Violations** | PII (Personally Identifiable Information), doxxing. |
| **8** | **Prompt Attacks** | Jailbreaks, prompt injections, adversarial inputs. |

---

## 🏗️ Architecture

Sentinel uses a **Dual-Rail Guardrail System**:

*   **Rail A (Input Guard)**: Detects `Prompt Attacks` (Category 8) before they reach the core LLM.
*   **Rail B (Output/Policy Guard)**: Detects policy violations (Categories 1-7) in both user input and model output.

---

## 📊 Dataset

The Sentinel-SLM dataset is a massive aggregation of SOTA open-source datasets, unified into a single taxonomy.

**Total Size:** ~1.67 Million Samples

| Source | Samples | Primary Focus | Language |
| :--- | :--- | :--- | :--- |
| **KoalaAI (Multilingual)** | 1.46M | Safety, Hate, Sexual | Multilingual (User Prompts) |
| **Civil Comments** | 180k | Harassment, Toxicity | English (Social Media) |
| **BeaverTails** | 27k | Policy Alignment (Red Teaming) | English (QA Pairs) |
| **MultiJail** | 3.8k | Prompt Injections / Jailbreaks | 9 Languages (Ar, Zh, It, etc.) |
| **JailbreakBench** | 200 | Sophisticated Jailbreaks | English |

---

## 🚀 Usage

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/abdulmunimjemal/Sentinel-SLM.git
cd Sentinel-SLM

# Setup Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file with your keys (required for downloading certain datasets or synthetic generation):
```bash
cp .env.example .env
# Edit .env and add HF_TOKEN or OPENROUTER_API_KEY
```

### 3. Run Data Pipeline

The entire data preparation process is unified in `scripts/run_pipeline.py`.

```bash
# Download all datasets (including BeaverTails, KoalaAI)
python scripts/run_pipeline.py --download

# Process and merge into final taxonomy
python scripts/run_pipeline.py --process

# (Optional) Generate synthetic data for specific gaps
python scripts/run_pipeline.py --synthetic --count 100
```

**Final Output:** `data/processed/final_augmented_dataset.parquet`

---

## 📁 Project Structure

```
Sentinel-SLM/
├── data/                   # Data storage (Raw & Processed)
│   ├── raw/                # Original Parquet downloads
│   └── processed/          # Final Unified Dataset
├── docs/                   # Documentation & Research
├── notebooks/              # Interactive Jupyter Notebooks
├── scripts/                # CLI Entry points
│   └── run_pipeline.py     # Main Pipeline Script
├── src/                    # Source Code
│   └── sentinel/
│       ├── data/           # Data Loaders & Processors
│       └── utils/          # Taxonomy definitions
├── tests/                  # Unit tests
├── .env.example            # Environment template
└── README.md               # This file
```

---

## 🤝 Contributing

This project is built for the open-source community. We prioritize:
1.  **Code Quality**: Type hints, docstrings, and modular design.
2.  **Safety**: Rigorous handling of sensitive content (red-teaming).

---

**Author**: Abdulmunim Jundur Rahman
**Status**: Data Preparation Complete / Ready for Training (Phase 2)

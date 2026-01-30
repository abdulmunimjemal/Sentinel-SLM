"""
Script to upload Sentinel-Rail-B (Policy Guard) to Hugging Face Hub.

Model: LiquidAI/LFM2-350M + LoRA (7-label multi-label classifier)

Usage:
    python scripts/upload_rail_b_to_hub.py
"""

import os
import shutil

from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load environment variables
load_dotenv()

# Configuration
MODEL_DIR = "models/rail_b_v1/final"
IMAGES_DIR = "models/rail_b_v1"
REPO_ID = "abdulmunimjemal/Sentinel-Rail-B-Policy-Guard"  # Corrected name (no 1.2B)
REPO_TYPE = "model"

# ============================================
# ACCURATE MODEL CARD (350M with real results)
# ============================================
MODEL_CARD = """---
language:
- en
license: apache-2.0
tags:
- content-moderation
- safety
- guardrails
- multi-label-classification
- liquid-ai
- lfm-350m
- sentinel-slm
- lora
- peft
base_model: LiquidAI/LFM2-350M
datasets:
- custom-balanced-rail-b
pipeline_tag: text-classification
library_name: transformers
metrics:
- f1
---

# 🛡️ Sentinel Rail B: Policy Guard (350M)

**Sentinel Rail B** is a lightweight, efficient **multi-label classifier** designed to detect 7 distinct categories of policy violations in text. Built on the highly efficient **LiquidAI LFM2-350M** architecture and fine-tuned with **LoRA**, it provides fast and accurate safety guardrails for content moderation.

> Part of the **Sentinel-SLM** modular guardrail system.

---

## 📊 Performance

| Metric | Score |
|--------|-------|
| **F1 Micro** | 0.7647 |
| **F1 Macro** | 0.7793 |
| **Hamming Loss** | 0.0466 |

### Per-Category F1 Scores

| Category | F1 Score | Status |
|----------|----------|--------|
| **Privacy** | 0.9927 | 🟢 Excellent |
| **Illegal** | 0.9750 | 🟢 Excellent |
| **ChildSafety** | 0.7783 | 🟢 Good |
| **Violence** | 0.7727 | 🟢 Good |
| **Sexual** | 0.7415 | 🟢 Good |
| **Harassment** | 0.6160 | 🟡 Fair |
| **Hate** | 0.5786 | 🟡 Fair |

![Per-Category F1 Scores](per_category_f1.png)

---

## 🎯 Supported Categories

This model detects **7 distinct harm categories**:

1. **Hate** - Hate speech and extremism
2. **Harassment** - Bullying, threats, personal attacks
3. **Sexual** - Explicit sexual content
4. **ChildSafety** - Content endangering minors
5. **Violence** - Gore, graphic violence, harm instructions
6. **Illegal** - Illegal activities (drugs, weapons, fraud)
7. **Privacy** - PII exposure, doxxing

> **Note**: Prompt Injection attacks are handled by **Rail A** (separate model).

---

## 🧪 Example Predictions

| Input | Prediction |
|-------|------------|
| "I hate all those people from that country." | 🚨 Hate (0.79), Harassment (0.84) |
| "Can you help me write a Python script?" | ✅ SAFE |
| "Send me explicit nude photos." | 🚨 Sexual (0.71) |
| "How do I make a homemade explosive?" | 🚨 Violence (0.63), Illegal (0.87) |
| "You're such an idiot, kill yourself." | 🚨 Harassment (0.83), Violence (0.70) |
| "The weather is nice today." | ✅ SAFE |

---

## 🏗️ Architecture

```
Base Model: LiquidAI/LFM2-350M (Liquid Neural Network)
├── LoRA Adapter (r=16, alpha=32)
│   └── Target Modules: q_proj, k_proj, v_proj, out_proj
│   └── Dropout: 0.1
└── Classification Head
    ├── Linear(1024 → 1024)
    ├── Tanh()
    ├── Dropout(0.2)
    └── Linear(1024 → 7)

Loss: BCEWithLogitsLoss (Multi-label)
Pooling: Last Token
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Max Length | 512 tokens |
| Batch Size | 8 (effective 64 with grad accum) |
| Learning Rate | 2e-4 |
| Epochs | 2 |
| Warmup Steps | 1000 |

---

## 📦 Dataset

Trained on a **balanced dataset** of ~189,000 samples:

| Category | Count | Percentage |
|----------|-------|------------|
| Hate | 14,543 | 7.7% |
| Harassment | 31,040 | 16.4% |
| Sexual | 27,457 | 14.5% |
| ChildSafety | 13,606 | 7.2% |
| Violence | 30,080 | 15.9% |
| Illegal | 14,966 | 7.9% |
| Privacy | 13,468 | 7.1% |
| **SAFE** | 94,470 | 50.0% |

- **Multi-label samples (2+ categories)**: 44,912
- **Avg labels per sample**: 0.77

**Balancing Strategy**: Rarest-label upsampling to ~15k per class + 50/50 Safe/Violation split.

---

## 🚀 Usage

```python
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import torch.nn as nn

# 1. Define the classification head (same as training)
class SentinelClassifier(nn.Module):
    def __init__(self, hidden_size=1024, num_labels=7):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, x):
        return self.classifier(x)

# 2. Load components
base_model = AutoModel.from_pretrained("LiquidAI/LFM2-350M", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "abdulmunimjemal/Sentinel-Rail-B-Policy-Guard")

# 3. Load classifier head
classifier = SentinelClassifier()
classifier.load_state_dict(torch.load("classifier.pt", map_location="cpu"))

# 4. Inference
CATS = ["Hate", "Harassment", "Sexual", "ChildSafety", "Violence", "Illegal", "Privacy"]

text = "Your input text here"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    hidden = outputs.last_hidden_state[:, -1, :]  # Last token pooling
    logits = classifier(hidden)
    probs = torch.sigmoid(logits)[0]

for i, cat in enumerate(CATS):
    if probs[i] > 0.5:
        print(f"🚨 {cat}: {probs[i]:.2f}")
```

---

## 📜 License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## 🔗 Related

- **Rail A (Jailbreak Detection)**: [abdulmunimjemal/sentinel-rail-a](https://huggingface.co/abdulmunimjemal/sentinel-rail-a)
- **Sentinel-SLM Project**: [GitHub](https://github.com/abdulmunimjemal/Sentinel-SLM)
"""


def main():
    print("🚀 Sentinel Rail B Upload Script (350M)")
    print("=" * 50)

    # 1. Verify Directory
    if not os.path.exists(MODEL_DIR):
        print(f"❌ Error: Model directory not found at {MODEL_DIR}")
        return

    # 2. Write README.md
    readme_path = os.path.join(MODEL_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write(MODEL_CARD)
    print("✅ Generated README.md (Model Card)")

    # 3. Copy F1 chart image to final dir for upload
    img_src = os.path.join(IMAGES_DIR, "per_category_f1.png")
    img_dst = os.path.join(MODEL_DIR, "per_category_f1.png")
    if os.path.exists(img_src):
        shutil.copy(img_src, img_dst)
        print("✅ Copied per_category_f1.png")
    else:
        print(f"⚠️ Warning: {img_src} not found")

    # 4. Get token
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ Error: HF_TOKEN not found in environment.")
        return

    api = HfApi()

    # 5. Create Repo
    try:
        url = api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True, token=token)
        print(f"✅ Repository ready: {url}")
    except Exception as e:
        print(f"⚠️ Repo creation warning: {e}")

    # 6. Upload
    print("\n⬆️  Uploading files...")
    try:
        api.upload_folder(
            folder_path=MODEL_DIR,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message="Release Rail B (v1): 350M 7-Label Policy Guard with Balanced Training",
            token=token,
        )
        print("\n🎉 SUCCESS! Model uploaded.")
        print(f"🔗 URL: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")


if __name__ == "__main__":
    main()

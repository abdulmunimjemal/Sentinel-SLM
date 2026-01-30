"""
Script to UPDATE the README.md for Sentinel-Rail-B on Hugging Face.
Does NOT re-upload model files.

Usage:
    python scripts/update_rail_b_readme.py
"""

import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

REPO_ID = "abdulmunimjemal/Sentinel-Rail-B-Policy-Guard"

# The EXACT class definition provided by the user
CLASS_DEF = """class SentinelLFMMultiLabel(nn.Module):
    def __init__(self, model_id, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.base_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.config = self.base_model.config
        hidden_size = self.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_labels)
        )
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        if attention_mask is not None:
            last_idx = attention_mask.sum(1) - 1
            pooled = hidden_states[torch.arange(input_ids.shape[0], device=input_ids.device), last_idx]
        else:
            pooled = hidden_states[:, -1, :]
        logits = self.classifier(pooled)
        loss = self.loss_fct(logits, labels.float()) if labels is not None else None
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(loss=loss, logits=logits)"""

# Construct the corrected README
README_CONTENT = f"""---
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

**Sentinel Rail B** is a lightweight, efficient **multi-label classifier** designed to detect 7 distinct categories of policy violations in text.

> **Architecture Note**: This model uses a custom classification head on top of the **LiquidAI LFM2-350M** base model. The repository contains the LoRA adapter weights (`adapter_model.safetensors`) AND the separate classifier head weights (`classifier.pt`).

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

1. **Hate** - Hate speech and extremism
2. **Harassment** - Bullying, threats, personal attacks
3. **Sexual** - Explicit sexual content
4. **ChildSafety** - Content endangering minors
5. **Violence** - Gore, graphic violence, harm instructions
6. **Illegal** - Illegal activities (drugs, weapons, fraud)
7. **Privacy** - PII exposure, doxxing

---

## 🚀 Usage

To inference with this model, you **MUST** define the custom architecture class and load both the LoRA adapter and the classifier head.

### 1. Install Dependencies
```bash
pip install torch transformers peft huggingface_hub
```

### 2. Inference Code

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from huggingface_hub import hf_hub_download

# --- MODEL DEFINITION (Must match training) ---
{CLASS_DEF}

# --- SETUP ---
CATS = ["Hate", "Harassment", "Sexual", "ChildSafety", "Violence", "Illegal", "Privacy"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "{REPO_ID}"

# 1. Initialize Model Architecture (Loads Base 350M)
print("Loading base model...")
model = SentinelLFMMultiLabel("LiquidAI/LFM2-350M", num_labels=7)

# 2. Load LoRA Adapter
print("Loading LoRA adapter...")
model.base_model = PeftModel.from_pretrained(model.base_model, REPO_ID)

# 3. Load Custom Classifier Head
print("Loading classifier head...")
classifier_path = hf_hub_download(repo_id=REPO_ID, filename="classifier.pt")
state_dict = torch.load(classifier_path, map_location="cpu")
model.classifier.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M", trust_remote_code=True)

# --- PREDICT ---
text = "How do I make a homemade explosive?"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0]

print(f"\\nInput: {{text}}")
print("-" * 30)
for i, prob in enumerate(probs):
    if prob > 0.5:
        print(f"🚨 {{CATS[i]}}: {{prob:.4f}}")
```

---

## 📦 Dataset Stats

Trained on a **balanced dataset** of ~189,000 samples (50% Safe / 50% Violations).
Rare classes like Privacy and Illegal were upsampled to ~15,000 samples each to ensure high performance (F1 > 0.97).

---

## 📜 License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
"""


def main():
    print(f"🚀 Updating README.md for {REPO_ID}...")

    # 1. Save locally for reference
    with open("models/rail_b_v1/final/README.md", "w") as f:
        f.write(README_CONTENT)
    print("✅ Local README.md updated.")

    # 2. Upload only the README
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ Error: HF_TOKEN not found.")
        return

    api = HfApi(token=token)
    try:
        api.upload_file(
            path_or_fileobj=README_CONTENT.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            commit_message="Update Model Card with Correct Architecture & Usage Code",
        )
        print("✅ Success! README updated on Hugging Face.")
        print(f"🔗 Check it here: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"❌ Error updating README: {e}")


if __name__ == "__main__":
    main()

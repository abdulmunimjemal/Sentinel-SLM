# Sentinel-SLM Architecture Documentation

## Overview

Sentinel-SLM implements a **Dual-Rail Guardrail System** designed to protect Large Language Model (LLM) deployments from both malicious inputs and policy-violating outputs. The architecture consists of two specialized components working in tandem:

- **Rail A (Input Guard)**: Binary classifier for prompt injection/jailbreak detection
- **Rail B (Policy Guard)**: Multi-label classifier for 8-category safety violations

---

## Model Registry

**Published Models (Hugging Face)**

- **Rail A (Prompt Attack Guard)**: `abdulmunimjemal/Sentinel-Rail-A-Prompt-Attack-Guard`
- **Rail B (Policy Guard)**: `abdulmunimjemal/Sentinel-Rail-B-Policy-Guard`

**Local Artifacts (this repo)**

- **Rail A**: `models/rail_a_v3/final/`
- **Rail B (training output)**: `models/rail_b_v1/final/`

---

## Why Liquid Models (LFM2)

Sentinel-SLM uses **LiquidAI LFM2** models because they are explicitly designed for **fast, low-latency inference and edge/on-device deployment** while preserving strong language understanding. The LFM2 family uses a modern transformer backbone with efficiency-oriented architectural choices that are well-suited to moderation workloads (short-to-medium prompts, frequent calls, tight latency budgets).

**Key reasons for the choice:**

- **Edge + on-device focus**: LFM2 models are designed to run efficiently across CPU/GPU/NPU environments, enabling deployment in latency-sensitive or privacy-constrained settings (see the Transformers LFM2 docs).
- **Efficiency-oriented architecture**: LFM2 uses **gated short-convolution + grouped-query attention** with **QK layer normalization**, combining efficient sequence modeling with strong attention performance (see the Transformers LFM2 docs).
- **Speed emphasis**: Liquid AI reports faster prefill and decode performance for LFM2 on CPU versus common open baselines, reinforcing its suitability for real-time guardrails (see the Liquid AI LFM2 blog).
- **Right-sized base**: For Rail A, the **350M** LFM2 model provides a strong performance/latency balance (model card lists ~354M params).

> **Note on latency**: Sentinel-SLM targets sub-50ms inference in production, but actual latency depends on hardware, batching, and deployment stack. Benchmark your own environment.

**References**
- https://huggingface.co/docs/transformers/model_doc/lfm2
- https://huggingface.co/LiquidAI/LFM2-350M
- https://www.liquid.ai/blog/introducing-lfm2-our-new-state-of-the-art-language-models

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Input                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │   Rail A (Input)     │  ← Prompt Injection Detection
            │   Binary Classifier  │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │   Decision: Safe?    │
            └──────────┬───────────┘
                       │
            ┌──────────┴───────────┐
            │                      │
          YES│                      │NO
            │                      │
            ▼                      ▼
    ┌──────────────┐      ┌──────────────┐
    │  Process     │      │   BLOCK      │
    │  with LLM    │      │   Request    │
    └──────┬───────┘      └──────────────┘
           │
           ▼
    ┌──────────────┐
    │  LLM Output  │
    └──────┬───────┘
           │
           ▼
    ┌──────────────────────┐
    │   Rail B (Output)    │  ← Policy Violation Detection
    │  Multi-label (7)     │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   Decision: Safe?     │
    └──────────┬───────────┘
               │
    ┌──────────┴───────────┐
    │                      │
  YES│                      │NO
    │                      │
    ▼                      ▼
┌──────────┐        ┌──────────┐
│  Return  │        │  BLOCK   │
│  Output  │        │  Output  │
└──────────┘        └──────────┘
```

---

## Rail A: Input Guard

### Purpose
Detect and block prompt injection attacks, jailbreaks, and adversarial inputs **before** they reach the core LLM.

### Architecture

#### Base Model
- **Model**: `LiquidAI/LFM2-350M`
- **Type**: Transformer-based encoder
- **Parameters**: ~355M (LFM2-350M model card)

#### Fine-tuning
- **Method**: LoRA (Low-Rank Adaptation)
- **Rank**: 16
- **Alpha**: 32
- **Target Modules**: Attention projection layers (`q_proj`, `k_proj`, `v_proj`, `out_proj`)
- **Trainable Parameters**: 1,015,808 (0.2857% of base model, from `notebooks/04_train_rail_a.ipynb`)

#### Classification Head
```
Input: [batch_size, seq_len, hidden_size]
  ↓
Last Token Pooling (with attention mask)
  ↓
Linear(hidden_size → hidden_size)
  ↓
Tanh()
  ↓
Dropout(0.1)
  ↓
Linear(hidden_size → 2)
  ↓
Output: [batch_size, 2] logits
```

### Output
- **Binary Classification**: `0` = Safe, `1` = Attack
- **Confidence**: Softmax probabilities for decision thresholding

### Performance
- **Accuracy**: 99.42%
- **F1 Score**: 99.45%
- **Precision**: 99.42%
- **Recall**: 99.83%

---

## Rail B: Policy Guard

### Purpose
Detect policy violations in both user inputs and LLM outputs across 8 safety categories.

### Architecture (Notebook-Defined)

#### Base Model
- **Model**: `LiquidAI/LFM2-350M`
- **Type**: Transformer-based encoder
- **Fine-tuning**: LoRA (r=16, alpha=32, dropout=0.1)

#### Classification Head
- **Multi-label Classification**: 7 categories (Prompt Attack excluded; handled by Rail A)
- **Output Format**: Binary logits per category with `BCEWithLogitsLoss`
- **Decision Logic**: Independent threshold per category (default 0.5)

### Categories

| ID | Category | Description |
|----|----------|-------------|
| 0  | Safe | No policy violations |
| 1  | Hate & Extremism | Hate speech, discrimination, extremism |
| 2  | Harassment | Bullying, severe toxicity, personal attacks |
| 3  | Sexual Content | NSFW, explicit sexual material |
| 4  | Child Safety | CSAM, exploitation, grooming |
| 5  | Violence | Gore, threats, self-harm |
| 6  | Illegal Activities | Drug trade, weapons, financial crimes |
| 7  | Privacy Violations | PII leaks, doxxing |
| 8  | Prompt Attacks | Jailbreaks, prompt injection |

**Note**: Category 8 overlaps with Rail A's domain. The Rail B training notebook excludes Prompt Attack by design and uses 7 labels.

---

## Data Pipeline

### Rail A Data Sources
1. **deepset/prompt-injections** - Labeled injection dataset
2. **JailbreakBench/JBB-Behaviors** - Jailbreak behavior goals
3. **TrustAIRLab/in-the-wild-jailbreak-prompts** - Real-world prompts
4. **Simsonsun/JailbreakPrompts** - Curated jailbreak prompts
5. **yanismiraoui/prompt_injections** - Additional samples

**Safe Samples**: Balanced using Dolly-15k and Alpaca datasets.

### Rail B Data Sources
1. **KoalaAI/Text-Moderation-Multilingual** - 1.46M samples (multilingual)
2. **Civil Comments (Jigsaw)** - 180k samples (harassment, toxicity)
3. **PKU-Alignment/BeaverTails** - 27k samples (policy violations)
4. **DAMO-NLP-SG/MultiJail** - 3.8k samples (multilingual jailbreaks)
5. **JailbreakBench** - 200 samples (sophisticated jailbreaks)

**Total**: ~1.67M samples across 8 categories

---

## Inference Pipeline

### Step 1: Input Preprocessing
```python
# Tokenize input
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
```

### Step 2: Rail A Classification
```python
# Load Rail A model
rail_a_model = load_rail_a_model("models/rail_a_v3/final")

# Classify
with torch.no_grad():
    outputs = rail_a_model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    prediction = torch.argmax(logits, dim=-1).item()

if prediction == 1:  # Attack detected
    return {"blocked": True, "reason": "prompt_injection"}
```

### Step 3: LLM Processing (if Rail A passes)
```python
if prediction == 0:  # Safe
    llm_output = llm.generate(inputs)
```

### Step 4: Rail B Classification
```python
# Load Rail B model (when available)
rail_b_model = load_rail_b_model("models/rail_b_v1/final")

# Classify output
with torch.no_grad():
    outputs = rail_b_model(**llm_outputs)
    category_logits = outputs.logits  # [batch, 8]

    # Apply thresholds per category
    violations = (torch.sigmoid(category_logits) > thresholds).cpu().numpy()

if violations.any():
    return {"blocked": True, "violations": violations}
else:
    return {"blocked": False, "output": llm_output}
```

---

## Model Deployment

### File Structure
```
models/
├── rail_a_v3/
│   └── final/
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       ├── classifier.pt
│       ├── tokenizer.json
│       └── config.json
└── rail_b_v1/  (future)
    └── final/
        └── ...
```

### Loading Models
```python
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import torch.nn as nn

# Load base model
base_model = AutoModel.from_pretrained("LiquidAI/LFM2-350M")
tokenizer = AutoTokenizer.from_pretrained("models/rail_a_v3/final")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "models/rail_a_v3/final")

# Load classification head
classifier = nn.Sequential(...)
classifier.load_state_dict(torch.load("models/rail_a_v3/final/classifier.pt"))
model.classifier = classifier
```

---

## Performance Characteristics

### Rail A
- **Inference Latency**: <50ms per sample
- **Throughput**: ~20 samples/second
- **Memory**: ~2 GB RAM
- **Model Size**: ~707 MB total

### Rail B (Estimated)
- **Inference Latency**: <100ms per sample
- **Throughput**: ~10 samples/second
- **Memory**: ~4-6 GB RAM
- **Model Size**: ~2-3 GB total

---

## Security Considerations

### Threat Model
1. **Prompt Injection**: Direct attempts to override system instructions
2. **Jailbreaks**: Sophisticated techniques to bypass safety filters
3. **Indirect Injection**: Attack patterns embedded in seemingly benign content
4. **Multilingual Attacks**: Attacks in non-English languages

### Defense Strategy
1. **Input Sanitization**: Rail A filters malicious inputs
2. **Output Validation**: Rail B validates LLM outputs
3. **Multi-layer Defense**: Redundant checks across both rails
4. **Continuous Monitoring**: Track attack patterns and update models

---

## Future Enhancements

### Short-term
- [ ] Complete Rail B training
- [ ] Multilingual evaluation suite
- [ ] Production deployment guide
- [ ] API server implementation

### Long-term
- [ ] Ensemble methods for improved robustness
- [ ] Adversarial training pipeline
- [ ] Active learning for continuous improvement
- [ ] Model compression for edge deployment

---

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Llama Guard](https://github.com/facebookresearch/PurpleLlama)
- [MLCommon Safety Benchmarks](https://github.com/mlcommons)


---

[← Previous: Introduction](00_introduction.md) | [Next: Installation & Usage →](02_installation_usage.md)

# Architecture

## Overview

Sentinel-SLM implements a **Dual-Rail Guardrail System** designed to protect Large Language Model (LLM) deployments from both malicious inputs and policy-violating outputs. The architecture consists of two specialized components working in tandem:

- **Rail A (Input Guard)**: Binary classifier for prompt injection/jailbreak detection.
- **Rail B (Policy Guard)**: Multi-label classifier for 8-category safety violations.

## model Registry

### Published Models (Hugging Face)
- **Rail A (Prompt Attack Guard)**: `abdulmunimjemal/Sentinel-Rail-A-Prompt-Attack-Guard`
- **Rail B (Policy Guard)**: `abdulmunimjemal/Sentinel-Rail-B-Policy-Guard`

### Local Artifacts
- **Rail A**: `models/rail_a_v3/final/`
- **Rail B**: `models/rail_b_v1/final/`

---

## System Architecture

Sentinel-SLM sits between your user and your LLM. It acts as a bidirectional firewall.

```mermaid
flowchart TD
    User([User Input]) --> RailA{Rail A\nInput Guard}

    RailA -- "🚨 Attack Detected" --> Block1[Block Request\nReturn 403]
    RailA -- "✅ Safe" --> LLM[Backbone LLM\n(e.g., Llama/GPT)]

    LLM --> RailB{Rail B\nPolicy Guard}

    RailB -- "🚨 Violation Detected" --> Block2[Block Output\nReturn Safety Message]
    RailB -- "✅ Safe" --> Response([Return Response])

    style RailA fill:#ff9999,stroke:#333,stroke-width:2px
    style RailB fill:#ff9999,stroke:#333,stroke-width:2px
    style Block1 fill:#ffcccc,stroke:#d62728,stroke-dasharray: 5 5
    style Block2 fill:#ffcccc,stroke:#d62728,stroke-dasharray: 5 5
    style LLM fill:#e1d5e7,stroke:#9673a6,stroke-width:2px
```

---

## Rail A: Input Guard

### Purpose
Detect and block prompt injection attacks, jailbreaks, and adversarial inputs **before** they reach the core LLM.

### Model Internals
Rail A is a 350M parameter transformer adapted with LoRA.

```mermaid
graph LR
    Input[Input Text] --> Tokenizer
    Tokenizer --> Base[LFM2-350M\nBase Model]
    Base --> LoRA[LoRA Adapters\n(Rank 16)]
    LoRA --> Head[Classification Head\n(Linear -> Tanh -> Linear)]

    Head --> Output{Output}
    Output --> Safe("Safe (0)")
    Output --> Attack("Attack (1)")

    style Base fill:#dae8fc,stroke:#6c8ebf
    style LoRA fill:#d5e8d4,stroke:#82b366
    style Head fill:#ffe6cc,stroke:#d79b00
```

### Performance
| Metric | Score | Note |
| :--- | :--- | :--- |
| **Accuracy** | **99.42%** | High reliability on benchmark attacks. |
| **Recall** | **99.83%** | Extremely low False Negative rate (missed attacks). |
| **Latency** | **<50ms** | CPU-friendly inference. |

---

## Rail B: Policy Guard

### Purpose
Detect policy violations in both user inputs and LLM outputs across 8 safety categories.

### Architecture
Rail B extends the same efficient 350M base with a **Multi-Label Classification Head**. Unlike Rail A (which is binary), Rail B outputs independent probabilities for 7 distinct categories.

> [!NOTE]
> **Why Multi-Label?**
> A single message can contain multiple violations (e.g., "Hate Speech" AND "Violence"). Our architecture detects all applicable tags simultaneously.

### Categories & Thresholds
Each category works independently. If *any* category exceeds its threshold (default 0.5), the content is flagged.

| ID | Category | Description |
|:---|:---|:---|
| 1 | **Hate & Extremism** | Hate speech, discrimination, extremism |
| 2 | **Harassment** | Bullying, severe toxicity, personal attacks |
| 3 | **Sexual Content** | NSFW, explicit sexual material |
| 4 | **Child Safety** | CSAM, exploitation, grooming |
| 5 | **Violence** | Gore, threats, self-harm |
| 6 | **Illegal Activities** | Drug trade, weapons, financial crimes |
| 7 | **Privacy Violations** | PII leaks, doxxing |

*(Category 8 "Prompt Attack" is handled exclusively by Rail A)*

---

## Why Liquid Models (LFM2)?

Sentinel-SLM uses **LiquidAI LFM2-350M** models.

1.  **Edge-Native Efficiency**: Designed for efficient inference on CPU/NPU, making it cheaper to run than Llama-3-8B.
2.  **Hybrid Architecture**: Uses gated short-convolutions + attention, providing a sweet spot between transformer quality and RNN speed.
3.  **Performance**: Outperforms similar-sized BERT models on language understanding tasks.

---

[← Previous: Introduction](00_introduction.md) | [Next: Installation & Usage →](02_installation_usage.md)

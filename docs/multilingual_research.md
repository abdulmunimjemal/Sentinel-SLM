# SOTA Multilingual Dataset Research (2025/2026)

This document summarizes high-quality, multilingual State-Of-The-Art (SOTA) datasets identified for expanding Sentinel-SLM's coverage beyond English.

## 1. Top Recommendations

### 🥇 KoalaAI/Text-Moderation-Multilingual
- **Use Case:** Core Policy Training (Rail B)
- **Size:** ~1.62M samples
- **Languages:** Multilingual (Aggregated from OpenAI, ifmain, etc.)
- **Labels:** 
    - `S` = Sexual
    - `H` = Hate
    - `V` = Violence
    - `HR` = Harassment
    - `SH` = Self-harm
    - `S3` = Sexual/Minors (Maps to **Child Safety**)
- **Pros:** Massive size, clean integer labels, directly maps to 5/8 of our categories. **Apache 2.0 License.**
- **URL:** [HuggingFace ID: `KoalaAI/Text-Moderation-Multilingual`](https://huggingface.co/datasets/KoalaAI/Text-Moderation-Multilingual)

### 🥈 ToxicityPrompts/PolyGuardMix (2025)
- **Use Case:** Core Policy Training (Rail B)
- **Size:** 1.91M samples
- **Languages:** 17 languages (Chinese, Czech, English, Hindi, etc.)
- **Description:** A massively multilingual safety training corpus released in 2025.
- **Pros:** True multilingual diversity, very recent.
- **URL:** [HuggingFace ID: `ToxicityPrompts/PolyGuardMix`](https://huggingface.co/datasets/ToxicityPrompts/PolyGuardMix)

### 🥉 DAMO-NLP-SG/MultiJail
- **Use Case:** Prompt Attack Evaluation (Rail A)
- **Size:** ~315 core prompts x 9 languages (2.8k total)
- **Languages:** Chinese, Italian, Vietnamese, Arabic, Korean, Thai, Bengali, Swahili, Javanese.
- **Description:** English unsafe prompts manually translated to test jailbreak resistance across languages.
- **Pros:** Critical for testing "Prompt Attack" robustness in non-English contexts.
- **URL:** [HuggingFace ID: `DAMO-NLP-SG/MultiJail`](https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail)

## 2. Other Candidates

| Dataset | Size | Focus | Notes |
| :--- | :--- | :--- | :--- |
| `textdetox/multilingual_toxicity_dataset` | Varies | Toxicity | Covers 10 languages including Russian, Ukrainian, Hindi. |
| `X-Hate 999` | Low | Hate Speech | Translated tweets in 6 languages. Good for evaluation. |
| `Aya RedTeaming` | ~7k | Alignment | 8 languages. Good for general safety alignment. |

## 3. Integration Plan (Proposed)

For **Phase 2 (Training)**, we strongly recommend integrating **KoalaAI/Text-Moderation-Multilingual**.

**Mapping Strategy:**
- `S` -> **3: Sexual**
- `H` -> **1: Hate**
- `V` -> **5: Violence**
- `HR` -> **2: Harassment**
- `S3` -> **4: Child Safety** (Critical boost for this category)
- `SH` -> **5: Violence** (Self-harm is often grouped here)

This single addition would instantly make Sentinel-SLM a robust multilingual system.

# Data & Taxonomy

This document details the Sentinel-SLM dataset, including the taxonomy (label definitions), statistics, and the specific subsets used for training Rail A and Rail B.

## Dataset Summary

The **Sentinel-SLM Dataset** is a large-scale, multilingual safety and content moderation dataset aggregating over **1.67 million** samples from state-of-the-art open sources, unified under a rigid **8-category taxonomy**.

It features a "Dual-Rail" design:
-   **Rail A (Input)**: Detection of Jailbreaks and Prompt Injections.
-   **Rail B (Policy)**: Detection of Hate, Harassment, Sexual Content, Violence, etc.

**Total Pool Size:** 1,671,001 rows
**Languages:** 20+ (Balanced spread across En, Ru, Uk, Ja, Zh, De, Es, It, Fr)

### Rail-Specific Datasets
The total pool is filtered and processed differently for each Rail:

#### Rail A (Input Guard)
- **Focus**: Prompt Injection & Jailbreaks
- **Total Training Samples**: 7,782
- **Composition**:
  - **Attack (50%)**: 3,896 samples (JailbreakBench, Deepset, MultiJail, etc.)
  - **Safe (50%)**: 3,886 samples (Dolly-15k, Alpaca)
- **Source**: `data/processed/rail_a_clean.parquet`

#### Rail B (Policy Guard)
- **Focus**: Content Policy Violation (Hate, Violence, etc.)
- **Total Training Samples**: 188,940
- **Composition**:
  - **Violations (50%)**: ~94k samples (Upsampled rare classes like Privacy/Illegal)
  - **Safe (50%)**: ~94k samples (Matched subset from KoalaAI/CivilComments)
- **Source**: `data/processed/rail_b_full.parquet`

---

## Taxonomy (Labels)

Multi-label classification scheme (ids `0` to `8`):

| ID | Label | Description | Count (Pool) |
| :--- | :--- | :--- | :--- |
| **0** | **Safe** | Benign content. | ~1.0M |
| **1** | **Hate & Extremism** | Hate speech, slurs, discrimination. | ~43k |
| **2** | **Harassment** | Bullying, severe toxicity, insults. | ~407k |
| **3** | **Sexual Content** | NSFW, pornography, sexual explicit. | ~138k |
| **4** | **Child Safety** | CSAM, exploitation, grooming. | ~14k |
| **5** | **Violence** | Gore, physical threats, self-harm. | ~83k |
| **6** | **Illegal Activities** | Weapons, drugs, financial crime. | ~4k |
| **7** | **Privacy Violations** | PII, doxxing. | ~1.4k |
| **8** | **Prompt Attacks** | Jailbreaks, injections, adversarial. | ~3.8k |

---

## Source Data

This dataset is a harmonization of the following sources:

1.  **KoalaAI/Text-Moderation-Multilingual**: Backbone of multilingual safety (~1.46M).
2.  **Civil Comments (Jigsaw)**: Deep coverage of harassment (~180k).
3.  **PKU-Alignment/BeaverTails**: Specific policy violations (~27k).
4.  **DAMO-NLP-SG/MultiJail**: Multilingual Jailbreak prompts (~3.8k).
5.  **JailbreakBench**: SOTA sophisticated jailbreaks (~200).

---

## Considerations

### Bias & Limitations
-   **Synthetic Origins**: A significant portion of the data is artificially generated, which may propagate bias.
-   **Western Bias**: Definitions of "Hate" align with US-centric standards.

### Intended Use
-   Training Content Moderation Classifiers.
-   Training Guardrail Models.
-   **NOT** for training generative models to produce harmful content.

---

[← Previous: Installation & Usage](02_installation_usage.md) | [Next: Training Results →](04_training_results.md)

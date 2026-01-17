# Dataset Card for Sentinel-SLM Dataset

## Dataset Summary

The **Sentinel-SLM Dataset** is a large-scale, multilingual safety and content moderation dataset designed to train Small Language Models (SLMs) for guardrailing applications. It aggregates over **1.67 million** samples from state-of-the-art open sources, unified under a rigid **8-category taxonomy**.

It features a "Dual-Rail" design:
-   **Rail A (Input)**: Detection of Jailbreaks and Prompt Injections.
-   **Rail B (Policy)**: Detection of Hate, Harassment, Sexual Content, Violence, etc.

**Total Size:** 1,671,001 rows
**Languages:** 20+ (Balanced spread across En, Ru, Uk, Ja, Zh, De, Es, It, Fr)
**License:** Aggregate (See Source Data licenses)

---

## Taxonomy (Labels)

Multi-label classification scheme (ids `0` to `8`):

| ID | Label | Description | Count (approx) |
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

## Dataset Structure

### Data Fields

-   **`text`** (string): The user input or model response to be classified.
-   **`labels`** (List[int]): A list of applicable category IDs (e.g., `[1, 5]`).
-   **`source`** (string): The identifier of the original dataset (e.g., `koala_multilingual`, `civil_comments`).
-   **`lang`** (string): (Enriched Version only) The ISO-639-1 language code predicted by FastText (e.g., `en`, `zh`, `ru`).

### Statistics

-   **Input Length**: Mean ~123 characters. Tail extends to 4k+.
-   **Multilingualism** (Enriched Analysis):
    -   **English**: 486,078 (~29%)
    -   **Russian**: 85,976 (~5.1%)
    -   **Ukrainian**: 78,868 (~4.7%)
    -   **Japanese**: 77,437 (~4.6%)
    -   **Chinese**: 77,428 (~4.6%)
    -   **German**: 75,967 (~4.5%)
    -   **Spanish**: 74,908
    -   **Italian**: 73,274
    -   **Korean**: 73,247
    -   **French**: 72,999

---

## Source Data

This dataset is a harmonization of the following sources:

1.  **KoalaAI/Text-Moderation-Multilingual**:
    -   *Role*: The backbone of multilingual safety (Hate, Sexual, Violence).
    -   *Contribution*: ~1.46M samples.
2.  **Civil Comments (Jigsaw)**:
    -   *Role*: Deep coverage of harassment, toxicity, and insults in English.
    -   *Contribution*: ~180k samples.
3.  **PKU-Alignment/BeaverTails**:
    -   *Role*: Question-Answering pairs for specific policy violations (Child Safety, Illegal).
    -   *Contribution*: ~27k samples.
4.  **DAMO-NLP-SG/MultiJail**:
    -   *Role*: Multilingual Jailbreak prompts (9 languages).
    -   *Contribution*: ~3.8k samples.
5.  **JailbreakBench**:
    -   *Role*: SOTA sophisticated jailbreaks (English).
    -   *Contribution*: ~200 samples.

---

## Considerations within the intended use

### Bias & Limitations
-   **Synthetic Origins**: A significant portion of the data (KoalaAI) is synthetically generated or distilled from larger models, which may propagate biases inherent in those teachers (e.g., GPT-4, Llama).
-   **Western Bias in Definitions**: The definitions of "Hate Speech" or "Harassment" align with US-centric online safety standards and may vary across cultures represented in the multilingual data.
-   **Label Noise**: Automated mapping from source taxonomies to the Sentinel 8-class taxonomy may introduce noise.

### Intended Use
-   Training Content Moderation Classifiers.
-   Training Guardrail Models (Input/Output filtering).
-   Red-teaming LLMs.

### Misuse
-   This dataset contains high-toxicity and harmful examples. It should **NOT** be used to train generative models to *produce* such content, explicitly only for *detection*.

---

## Citation

If you use this dataset, please cite the underlying sources (KoalaAI, Jigsaw, PKU-Alignment, DAMO-NLP) and the Sentinel-SLM project.

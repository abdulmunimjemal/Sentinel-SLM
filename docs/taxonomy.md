# Data Dictionary & Taxonomy

## Core Policy Taxonomy (8 Categories)
Sentinel-SLM classifies content into the following 8 categories. This taxonomy maps to standard safety guidelines (e.g., Llama Guard, MLCommon).

| ID | Category Name | Description | Source Datasets |
| :--- | :--- | :--- | :--- |
| **0** | **Safe** | Content that does not violate any policy. | All (Negative examples) |
| **1** | **Hate & Extremism** | Hate speech, discriminatory language, promotion of extremist ideologies or terrorism. | Jigsaw, Civil Comments, BeaverTails |
| **2** | **Harassment & Bullying** | Targeted insults, threats, severe toxicity towards individuals. | Civil Comments, Jigsaw |
| **3** | **Sexual Content** | Explicit sexual descriptions, adult content, NSFW material. | BeaverTails, Civil Comments (subset) |
| **4** | **Child Safety & Exploitation** | Content depicting or promoting abuse/harm towards minors. **(Critical)** | BeaverTails, Synthetic |
| **5** | **Violence & Gore** | Graphic violence, self-harm, cruelty, gore. | BeaverTails, Jigsaw, DUTA |
| **6** | **Illegal Activities** | Promotion of crimes, drug use, weapons dealing, financial fraud. | BeaverTails, Synthetic |
| **7** | **Privacy Violations** | Doxing, PII leaks, unauthorized sharing of personal info. | BeaverTails, Synthetic |
| **8** | **Prompt Attacks** | Jailbreaks, prompt injection, adversarial attempts to bypass safety filters. | JailbreakBench, Synthetic |

## Dataset Schema
The final dataset (`data/processed/final_augmented_dataset.parquet`) follows this schema:

- **text** (`string`): The input content to be moderated.
- **labels** (`list[int]`): A list of category IDs (0-8) that apply to the text.
    - **Rail A (Input Guard)**: Binary Classification (Safe=0, Attack=1). Uses source datasets mapped to PromptAttack.
    - **Rail B (Policy Guard)**: Multi-label Classification for Categories 1-7. (Category 8 is excluded).
    - Example: `[2, 5]` means Harassment AND Violence.
    - Example: `[0]` means Safe.
- **source** (`string`): Origin of the sample (e.g., "civil_comments", "deepset_injection").
- **lang** (`string`): Language code (e.g., "en", "es").

## Data Volume
As of Data Prep Phase 1:
- **Total Samples**: ~208,000
- **Distribution**: Heavily skewed towards `Safe` (~85%). Harassment and Violence are well-represented. Child Safety and Prompt Attacks use synthetic augmentation.

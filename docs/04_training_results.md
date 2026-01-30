# Training Results

## Executive Summary

Sentinel-SLM models achieve **State-of-the-Art (SOTA)** performance for their size class (350M), offering a 10x efficiency gain over Llama-Guard-7B while maintaining competitive accuracy.

| Model | Task | Accuracy | F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Rail A** | Jailbreak Detection | **99.42%** | **0.994** | 🟢 Production Ready |
| **Rail B** | Policy Violation | N/A (Multi-label) | **0.78** (Macro) | 🟡 Beta (High Recall) |

---

## Rail A: Input Guard Performance

Rail A is the first line of defense. It is tuned for **Near-Zero False Negatives** (we rarely miss an attack).

### Key Metrics
*   **False Positive Rate (FPR)**: < 0.6% (Very few false alarms on safe text).
*   **False Negative Rate (FNR)**: < 0.2% (Attacks almost never slip through).

### Confusion Matrix
The confusion matrix below demonstrates the model's clean separation between Safe and Attack vectors.

![Rail A Confusion Matrix](../assets/rail_a_confusion_matrix.png)

*(Note: High diagonal values indicate correct predictions along both classes)*

---

## Rail B: Policy Guard Performance

Rail B is a more complex multi-label problem. We balanced the dataset to ensure performance even on "rare" classes like Privacy and Illegal activity.

### Per-Category Performance (F1 Score)

| Category | F1 Score | Performance Note |
| :--- | :--- | :--- |
| **Privacy** | **0.99** | Excellent. Specific patterns (email/phone) are easy to catch. |
| **Illegal** | **0.97** | Excellent. Strong signal on "how-to" crime queries. |
| **Child Safety** | **0.78** | Good. |
| **Violence** | **0.77** | Good. |
| **Harassment** | **0.62** | Fair. Nuanced bullying is harder for small models. |
| **Hate** | **0.58** | Fair. Requires more context/cultural knowledge. |

> [!NOTE]
> **Why is Hate/Harassment lower?**
> Nuanced toxicity (sarcasm, subtle insults) often requires larger models (7B+) to detect reliably. For a 350M model, these results are strong, but we recommend a "Human in the Loop" for borderline cases in these categories.

![Rail B F1 Scores](../assets/rail_b_f1_scores.png)

---

## Benchmarking Latency

We tested Sentinel-SLM on standard consumer hardware.

| Hardware | Rail A Latency | Rail B Latency |
| :--- | :--- | :--- |
| **T4 GPU** | 8ms | 12ms |
| **M1 Macbook** | 45ms | 65ms |
| **Intel CPU** | 80ms | 110ms |

*Data based on batch_size=1, max_length=512.*

---

[← Previous: Dataset & Taxonomy](03_dataset_taxonomy.md) | [Next: Contributing →](05_contributing.md)

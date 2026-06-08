# MedVox LatAm

### Benchmarking Speech Recognition Models for Medical Consultations in Latin American Spanish: A Comparative Evaluation with Fine-Tuning

This repository contains the evaluation results of speech-to-text (STT) models — also known as automatic speech recognition (ASR) — applied to audio recordings of medical consultations conducted in Spanish from Latin America (LatAm). We benchmark ten STT models (five open-source and five close-source) against human-derived gold-standard transcriptions, and we report the results of fine-tuning Whisper Large v3 on a domain-specific dataset.

---

## Methods

### Data sources

Dataset of ten YouTube videos depicting medical consultations in Spanish. A human transcriber produced verbatim transcriptions. These human transcriptions are the gold-standard reference against which all STT models are evaluated.

### Montreal Forced Aligner (MFA)

We used the Montreal Forced Aligner (MFA) to segment each full-length video into ~10-second audio chunks and to align each segment with the human transcription. One video (v00003) could not be processed by MFA and was excluded from fine-tuning. v00003 was retained as an external validation set.

Processing the nine eligible videos yielded **594 audio segments**. The mean segment duration was 10.7 seconds (SD: 2.52 seconds), and the aggregate duration of the dataset was 6,356.51 seconds (105.94 minutes).

### Data augmentation

For each of the 594 original chunks, four augmented variants were generated, yielding a final dataset of **2,970 audio samples** (594 original + 2,376 augmented). Augmentation was implemented using the *Audiomentations* library in Python under four regimes:

- **Noisy clean**
- **Multi-speaker**
- **Acoustic environment**
- **Broad combination** — a stack of perturbations including Gaussian noise, pitch shifting, temporal stretching, gain variation, temporal shifting, colored noise, and a 30% probability of synthetic reverberation.

### Benchmarked models

**Close-source (n = 5):** gpt-4o-transcribe; gpt-4o-mini-transcribe; gemini-2.5-pro; Eleven Labs (scribe_v2); Assembly AI.

**Open-source (n = 5):** Whisper Large; Whisper Large v3; Whisper Large v3 Turbo; Voxtral Mini 3B; Canary 1B v2.

### Evaluation metrics

Six metrics were computed against the human transcriptions:

- Word Error Rate (WER)
- Character Error Rate (CER)
- BLEU Score
- ROUGE-L
- BERT Score
- Semantic Similarity, computed via `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (the top sentence-transformer multilingual model at the time of the analysis).

### Fine-tuning

Whisper Large v3 was selected for fine-tuning. Training followed a **nine-fold leave-one-out (LOO) cross-validation design** across the nine eligible videos, with v00003 fully withheld for external validation.

**Fixed training hyperparameters**

- Batch size: 2
- Maximum epochs: 50
- Warm-up steps: 20
- Evaluation / checkpoint-saving interval: every 10 steps
- Gradient accumulation factor: 2

**Tuned hyperparameters**

- Learning rates: 1e-5, 5e-6, 2e-5
- Inference beam sizes: 3, 5 (default), 7
- Data augmentation: yes / no

LOO validation results were summarized as the mean and SD across the nine folds.

### Statistical analysis

The statistical analysis was descriptive. *A priori* hypothesis was that the fine-tuned model would demonstrate superior performance compared to the benchmarked models when applied to a single video (v00003).

Three analyses were performed:

1. **Baseline benchmarking.** The ten vanilla STT models (close-source and open-source) were applied to all ten videos. Performance metrics were summarized as means and standard deviations across the ten videos.
2. **LOO fine-tuning.** Performance metrics for the fine-tuned model were summarized as mean and SD across the nine folds, for each metric and for each hyperparameter configuration.
3. **External validation.** The ten vanilla STT models and the fine-tuned model were applied to the fully withheld video (v00003). Validation metrics were computed and models were ranked.

---

## Results

### 1. Vanilla STT models on the ten full-length videos

In the evaluation of the ten full-length videos, close-source models outperformed open-source models (Table 1). **Gemini-2.5-pro** emerged as the leading close-source model, ranking first in four of six metrics. Within the open-source group, **Whisper Large v3** consistently outperformed its counterparts.

### 2. Fine-tuned Whisper Large v3 — leave-one-out (LOO) internal validation

**None of the fine-tuning iterations outperformed the vanilla Whisper Large v3** (Table 2). Fine-tuning with data augmentation produced worse performance than fine-tuning without augmentation.

The configuration using **learning rate 5e-6, beam size 5, and no augmentation** achieved the metrics closest to the vanilla model — WER 19.7% (vs. 18.6% vanilla), CER 12.7% (vs. 11.6% vanilla), and Semantic Similarity 92.7% (vs. 93.1% vanilla) — and yielded the smallest cross-fold standard deviation across all six metrics. This configuration was selected to produce the final fine-tuned model.

### 3. External validation on the withheld video (v00003)

When the fine-tuned model was applied to v00003 (Table 3), and compared against the close-source models, the fine-tuned model did **not** outperform them on any metric; it ranked fourth out of six across five of the six metrics. In comparison to the other open-source models, however, the fine-tuned model ranked **first across all six metrics**.

---

## Tables

### Table 1. Performance of the vanilla models applied to the ten full-length videos.

Values are mean (SD) across the ten videos. Numbers in parentheses indicate the rank within each group (1 = best, 5 = worst).

**Close-source models**

| Model | WER Mean (rank) | WER SD (rank) | CER Mean (rank) | CER SD (rank) | BLEU Mean (rank) | BLEU SD (rank) | ROUGE-L Mean (rank) | ROUGE-L SD (rank) | BERT Mean (rank) | BERT SD (rank) | SemSim Mean (rank) | SemSim SD (rank) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-4o-transcribe | 0.297 (4) | 0.168 (4) | 0.224 (4) | 0.118 (4) | 0.603 (4) | 0.198 (4) | 0.807 (4) | 0.125 (4) | 0.925 (1) | 0.020 (1) | 0.956 (2) | 0.027 (2) |
| gpt-4o-mini-transcribe | 0.420 (5) | 0.303 (5) | 0.340 (5) | 0.246 (5) | 0.542 (5) | 0.245 (5) | 0.717 (5) | 0.213 (5) | 0.918 (4) | 0.036 (4) | 0.953 (3) | 0.040 (4) |
| gemini-2.5-pro | **0.140 (1)** | **0.058 (1)** | **0.090 (1)** | **0.042 (1)** | **0.780 (1)** | 0.074 (2) | **0.913 (1)** | **0.037 (1)** | 0.922 (2) | 0.029 (3) | **0.958 (1)** | 0.031 (3) |
| Eleven Labs (scribe_v2) | 0.170 (3) | 0.072 (3) | 0.118 (3) | 0.056 (3) | 0.742 (3) | 0.082 (3) | 0.896 (3) | 0.045 (2) | 0.883 (3) | 0.037 (5) | 0.933 (4) | 0.063 (5) |
| Assembly AI | 0.156 (2) | 0.067 (2) | 0.105 (2) | 0.050 (2) | 0.769 (2) | **0.073 (1)** | 0.897 (2) | 0.047 (3) | 0.922 (2) | 0.021 (2) | **0.958 (1)** | **0.027 (1)** |

**Open-source models**

| Model | WER Mean (rank) | WER SD (rank) | CER Mean (rank) | CER SD (rank) | BLEU Mean (rank) | BLEU SD (rank) | ROUGE-L Mean (rank) | ROUGE-L SD (rank) | BERT Mean (rank) | BERT SD (rank) | SemSim Mean (rank) | SemSim SD (rank) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Whisper Large | 0.578 (3) | 0.480 (5) | 0.402 (3) | 0.326 (4) | 0.456 (3) | 0.335 (5) | 0.560 (3) | 0.343 (5) | 0.893 (2) | 0.047 (3) | 0.940 (2) | **0.053 (1)** |
| Whisper Large v3 Turbo | 0.196 (2) | 0.091 (3) | 0.122 (2) | 0.064 (3) | 0.708 (2) | 0.106 (4) | 0.866 (2) | 0.065 (4) | 0.888 (3) | 0.044 (2) | 0.913 (3) | 0.083 (3) |
| **Whisper Large v3** | **0.186 (1)** | **0.079 (1)** | **0.118 (1)** | 0.057 (2) | **0.722 (1)** | 0.097 (3) | **0.874 (1)** | 0.057 (2) | **0.910 (1)** | **0.033 (1)** | **0.934 (1)** | 0.060 (2) |
| Voxtral Mini 3B | 1.546 (5) | 0.408 (4) | 1.172 (5) | 0.406 (5) | 0.029 (4) | 0.019 (2) | 0.118 (4) | 0.058 (3) | 0.662 (4) | 0.051 (4) | 0.777 (4) | 0.143 (5) |
| Canary 1B v2 | 0.956 (4) | 0.027 (2) | 0.864 (4) | 0.055 (1) | 0.006 (5) | **0.007 (1)** | 0.081 (5) | **0.045 (1)** | 0.636 (5) | 0.058 (5) | 0.697 (5) | 0.119 (4) |

*CER = Character Error Rate; SD = standard deviation; WER = Word Error Rate. For WER and CER, lower values indicate better performance; for BLEU Score, ROUGE-L, BERT Score, and Semantic Similarity, values closer to 1.0 indicate better performance. The numerical value enclosed in parentheses (1–5) signifies the ranking within each group of models (close-source and open-source), with 1 denoting the highest performance and 5 the lowest. Bolded values indicate the best mean or smallest SD within each group, column by column.*

---

### Table 2. Validation metrics during fine-tuning — internal leave-one-out (LOO) validation.

Each cell shows the mean (SD) across the nine LOO folds. **Bolded** values highlight the best performance metric or the smallest standard deviation, row by row. Video v00003 was not included in these results because it was not included during the fine-tuning process. The third fine-tuning configuration (LR = 5e-6, beam = 5, no augmentation) was selected for the final model.

| Metric | Baseline (Whisper Large v3) | LR = 1e-5, Beam = 5, Aug = Y | LR = 1e-5, Beam = 5, Aug = N | **LR = 5e-6, Beam = 5, Aug = N (selected)** | LR = 2e-5, Beam = 5, Aug = N | LR = 5e-6, Beam = 3, Aug = N | LR = 5e-6, Beam = 7, Aug = N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER | **0.1869** (0.0782) | 0.3004 (0.1233) | 0.2226 (0.1016) | 0.1970 (**0.0767**) | 0.3022 (0.1554) | 0.2169 (0.1264) | 0.2021 (0.0902) |
| CER | **0.1164** (0.0533) | 0.2192 (0.1068) | 0.1435 (0.0658) | 0.1277 (**0.0525**) | 0.2099 (0.1469) | 0.1496 (0.1037) | 0.1333 (0.0661) |
| BLEU Score | **0.7190** (0.0865) | 0.6029 (0.1389) | 0.6904 (0.1104) | 0.7152 (**0.0876**) | 0.5869 (0.1834) | 0.6956 (0.1343) | 0.7155 (0.0981) |
| ROUGE-L | **0.8706** (0.0533) | 0.7899 (0.0943) | 0.8468 (0.0727) | 0.8635 (**0.0550**) | 0.7763 (0.1214) | 0.8475 (0.0945) | 0.8606 (0.0632) |
| BERT Score | **0.9201** (0.0316) | 0.8845 (0.0508) | 0.9023 (0.0400) | 0.9091 (**0.0321**) | 0.8531 (0.0735) | 0.9030 (0.0431) | 0.9027 (0.0494) |
| Semantic Similarity | **0.9311** (0.0631) | 0.9257 (0.0577) | 0.9251 (0.0627) | 0.9271 (**0.0588**) | 0.8321 (0.1770) | 0.9294 (0.0626) | 0.9030 (0.0963) |

*CER = Character Error Rate; LR = learning rate; SD = standard deviation; WER = Word Error Rate. "Aug = Y/N" indicates whether the augmented dataset (n = 2,970) or the original dataset (n = 594) was used in fine-tuning. Results represent means and standard deviations across nine LOO folds (the fine-tuned model was applied to the held-out full-length video in each fold). For WER and CER, lower values indicate better performance; for BLEU Score, ROUGE-L, BERT Score, and Semantic Similarity, values closer to 1.0 indicate better performance.*

---

### Table 3. Performance of all models applied to the withheld video (v00003) only.

Video v00003 was excluded from all stages of model fine-tuning and serves as a fully external validation set. The fine-tuned model is shown twice — once compared against the close-source group, and once against the open-source group — for direct ranking within each.

**Close-source models (vs. fine-tuned Whisper Large v3)**

| Model | WER | CER | BLEU Score | ROUGE-L | BERT Score | Semantic Similarity |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-4o-transcribe | 0.6756 (6) | 0.6538 (6) | 0.1558 (5) | 0.4993 (6) | 0.9101 (2) | 0.9543 (3) |
| gpt-4o-mini-transcribe | 0.6728 (5) | 0.6465 (5) | 0.1547 (6) | 0.5041 (5) | **0.9121 (1)** | 0.9726 (2) |
| gemini-2.5-pro | **0.2078 (1)** | **0.1465 (1)** | **0.6999 (1)** | **0.8715 (1)** | 0.8608 (5) | 0.8866 (6) |
| Eleven Labs (scribe_v2) | 0.2190 (2) | 0.1515 (2) | 0.6764 (2) | 0.8570 (2) | 0.8504 (6) | 0.9175 (4) |
| Assembly AI | 0.2334 (3) | 0.1612 (3) | 0.6703 (3) | 0.8379 (3) | 0.9043 (3) | **0.9795 (1)** |
| Whisper Large v3 (fine-tuned) | 0.2933 (4) | 0.1967 (4) | 0.6309 (4) | 0.7966 (4) | 0.8913 (4) | 0.8987 (5) |

**Open-source models (vs. fine-tuned Whisper Large v3)**

| Model | WER | CER | BLEU Score | ROUGE-L | BERT Score | Semantic Similarity |
| --- | --- | --- | --- | --- | --- | --- |
| Whisper Large | 0.7818 (4) | 0.5751 (4) | 0.1903 (4) | 0.2541 (4) | 0.8639 (2) | 0.8328 (2) |
| Whisper Large v3 Turbo | 0.2996 (3) | 0.2120 (3) | 0.5778 (3) | 0.7912 (3) | 0.8309 (4) | 0.7185 (4) |
| Whisper Large v3 | 0.2963 (2) | 0.2049 (2) | 0.5805 (2) | 0.7931 (2) | 0.8420 (3) | 0.8202 (3) |
| Voxtral Mini 3B | 0.9863 (5) | 0.7021 (5) | 0.0016 (5) | 0.0665 (5) | 0.6046 (5) | 0.6215 (5) |
| Canary 1B v2 | 0.9995 (6) | 0.9048 (6) | 0.0000 (6) | 0.0019 (6) | 0.5380 (6) | 0.4810 (6) |
| **Whisper Large v3 (fine-tuned)** | **0.2933 (1)** | **0.1967 (1)** | **0.6309 (1)** | **0.7966 (1)** | **0.8913 (1)** | **0.8987 (1)** |

*CER = Character Error Rate; WER = Word Error Rate. For WER and CER, lower values indicate better performance; for BLEU Score, ROUGE-L, BERT Score, and Semantic Similarity, values closer to 1.0 indicate better performance. Bolded values highlight the best performance metric column-wise within each group. The numerical value enclosed in parentheses (1–6) signifies the ranking, with 1 denoting the highest and 6 the lowest.*

---

## Conclusions

Whisper Large v3 and its fine-tuned variant represent the most accurate **open-source** STT models identified for transcribing medical conversations in LatAm Spanish; **Gemini-2.5-pro** was the best-performing **close-source** model.

# MedVox LatAm
## Speech2Text Transcription of Latin American Spanish Medical Consultations

This evaluation assesses the performance of speech-to-text (Speech2Text) models, also known as automatic speech recognition (ASR), in transcribing audios of medical consultations conducted in Spanish from Latin America.


## Data and Methods:
- Ten videos (YouTube) depicting medical consultations conducted in Latin America (6 different countries).
- Human transcriptions served as the ground truth.
- Open-source Speech2Text models were tested with the tem videos.
- The base (vanilla) model which demonstrated the best performance, underwent fine-tuning.
- Fine-Tuning:
-   We utilized the Montreal Forced Aligner (MFA) to generate 10-second clips from the 9 videos that depicted medical consultations in Spanish from Latin America.
-   Data augmentation was employed to enhance the dataset.
-   The augmented data of 10-second clips were utilized in the fine-tuning process. Per 10-second clip, there were three files.
-   For internal validation, we used leave-one-out (LOO). In each iteration, the 10-second clips from eight videos were used to train the model, and the full-length video not used in training was witheld to compute the validation metrics.
-   The original human-made transcription served as the ground truth.
- Finally, the fine-tuned model was applied to one full video. The video that was not used at all during the training phase. We originally had 10 videos, nine were used during the training with LOO internal validation, whereas one video was never used and reserved for external validation.


## Conclusions:
The Whisper Large v3 model demonstrated good performance in transcribing videos depicting medical consultations in Spanish from Latin America. A fine-tuned version of Whisper Large v3 significantly enhanced the validation metrics compared to human transcriptions. Our mode, MedVox LatAm outperformed both open-source and close-source models.


## Results:

### Base models (open-source) applied to the ten videos -> the best-performing model will undergo fine-tuning.
| Model     | WER | CER | BLEU | ROUGE L | BERT | Cosine Similarity |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Whisper_large_v3 | 0.174104 | 0.116061 | 7.429227e-01 | 0.882841 | 0.905827 | 0.946515 |
| Whisper_large_v3_turbo | 0.197340 | 0.130502 | 7.308557e-01 | 0.872501 | 0.894323 | 0.939732 |
| Whisper_large | 0.604185 | 0.429750 | 4.260967e-01 | 0.533557 | 0.892381 | 0.923476 |
| Canary_1b_v2 | 0.956453 | 0.863769 | 6.070437e-03 | 0.080839 | 0.636031 | 0.696910 |
| Voxtral_Mini_3B_2507 | 0.994594 | 0.992461 | 2.708311e-43 | 0.013489 | 0.614435 | 0.546275 |

*sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2


### Base models (close-source) applied to the ten videos
| Model     | WER | CER | BLEU | ROUGE L | BERT | Cosine Similarity |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| GPT-4o-Transcribe | 0.355551 | 0.274422 | 5.95577e-01 | 0.790943 | 0.912620 | 0.946915 |
| GPT-4o-mini-Transcribe | 0.1972120 | 0.1475670 | 7.214053e-01 | 0.8676300 | 0.9095200 | 0.9484410 |
| Gemini-3-Flash-Preview | 0.197577 | 0.101842 | 7.84622e-01 | 0.907386 | 0.895780 | 0.957804 |

*sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2


### Fine tuning (Whisper Large V3 Fine-Tuned) -> LOO internal validation [mean (standard deviation) across the nine iterations]
| Learning Rate | WER | CER | BLEU | ROUGE L | BERT | Cosine Similarity* |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 1e-05 | 0.2262 (0.1088) | 0.1484 (0.0806) | 0.6871 (0.1147) | 0.8401 (0.0817) | 0.8952 (0.0467) | 0.9243 (0.0663) |
| 5e-06 | 0.2170 (0.1187) | 0.1489 (0.0988) | 0.6986 (0.1345) | 0.8500 (0.0853) | 0.9104 (0.0400) | 0.9089 (0.0955) |
| 2e-05 | 0.3225 (0.2054) | 0.2218 (0.1576) | 0.5799 (0.2007) | 0.7595 (0.1659) | 0.8486 (0.0798) | 0.8318 (0.2010) |

*sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2


# Speech2Text Transcription of Latin American Spanish Medical Consultations

This evaluation assesses the performance of speech-to-text (Speech2Text) models, also known as automatic speech recognition (ASR), in transcribing audios of medical consultations conducted in Spanish from Latin America.


## Data and Methods:
- 10 videos depicting medical consultations conducted in Latin America (6 different countries).
- Human transcriptions served as the ground truth.
- Open-source Speech2Text models were tested.
- The base (vanilla) model which demonstrated the best performance, underwent fine-tuning. We utilized the Montreal Forced Aligner (MFA) to generate 10-second clips from the 10 videos that depicted medical consultations in Spanish from Latin America. Data augmentation was employed to enhance the dataset (AddGaussianNoise, TimeStretch, PitchShift). The augmented data of 10-second clips were utilized in the fine-tuning process, which comprised training, validation, and testing phases in a 70-15-15 ratio. The original human-made transcription served as the ground truth. Finally, the fine-tuned model was applied to the full videos, similar to the base models.


## Results:

### Base models
| Model     | WER | CER | BLEU | ROUGE L | BERT | Cosine Similarity |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Whisper_large_v3 | 0.174104 | 0.116061 | 7.429227e-01 | 0.882841 | 0.905827 | 0.946515 |
| Whisper_large_v3_turbo | 0.197340 | 0.130502 | 7.308557e-01 | 0.872501 | 0.894323 | 0.939732 |
| Whisper_large | 0.604185 | 0.429750 | 4.260967e-01 | 0.533557 | 0.892381 | 0.923476 |
| Canary_1b_v2 | 0.956453 | 0.863769 | 6.070437e-03 | 0.080839 | 0.636031 | 0.696910 |
| Voxtral_Mini_3B_2507 | 0.994594 | 0.992461 | 2.708311e-43 | 0.013489 | 0.614435 | 0.546275 |

*sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

### Fine tuning
| Model | WER | CER | BLEU | ROUGE L | BERT | Cosine Similarity* |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Whisper Large V3 Fine-Tuned | 0.133642 | 0.081280 | 8.324975e-01 | 0.916292 | 0.951343 | 0.977602 |

*sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2


## Conclusions:
The Whisper Large v3 model demonstrated good performance in transcribing videos depicting medical consultations in Spanish from Latin America. A fine-tuned version of Whisper Large v3 significantly enhanced the validation metrics compared to human transcriptions.

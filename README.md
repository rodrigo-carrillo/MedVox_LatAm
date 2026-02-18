# Speech2Text Transcription of Latin American Spanish Medical Consultations

This evaluation assesses the performance of speech-to-text (Speech2Text) models, also known as automatic speech recognition (ASR), in transcribing audios of medical consultations conducted in Spanish from Latin America.

Data:
- 10 videos depicting medical consultations conducted in Latin America (6 different countries).

Methods:
- Human transcriptions served as the ground truth.
- Open-source Speech2Text models were tested.

Results:
| Model     | WER | CER | BLEU | ROUGE L | BERT | Cosine Similarity |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Whisper_large_v3 | 0.174104 | 0.116061 | 7.429227e-01 | 0.882841 | 0.905827 | 0.946515 |
| Whisper_large_v3_turbo | 0.197340 | 0.130502 | 7.308557e-01 | 0.872501 | 0.894323 | 0.939732 |
| Whisper_large | 0.604185 | 0.429750 | 4.260967e-01 | 0.533557 | 0.892381 | 0.923476 |
| Canary_1b_v2 | 0.956453 | 0.863769 | 6.070437e-03 | 0.080839 | 0.636031 | 0.696910 |
| Voxtral_Mini_3B_2507 | 0.994594 | 0.992461 | 2.708311e-43 | 0.013489 | 0.614435 | 0.546275 |

\*sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

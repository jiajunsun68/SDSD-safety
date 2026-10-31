# Cross-family check: Qwen2.5-7B-Instruct

Llama-3.1-8B-Instruct weights were removed locally, so the pre-registered fallback Qwen2.5-7B-Instruct is used. All models use their native chat template/EOS. XSTest string refusal is shown only as a diagnostic: it misses common fine-tuned refusal phrasings (e.g. "I strongly advise against..."), so **semantic refusal is the valid refusal column for this family** (Qwen3Guard). The idx6 attack is at ceiling for Base and cannot test an anchor-specific increase.

| Run | HB direct ASR % | XSTest safe semantic % | XSTest unsafe semantic % | XSTest safe string (diag.) % | GSM lm-eval % | GSM exact (diag.) % | idx6 ASR@1/4/16 % | length-hit % | dev normal refusal % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 6.00 | 6.00 | 82.50 | 4.00 | 94.50 | 68.00 | 93.12/99.87/100.00 | 29.17 | — |
| anchor | 10.50 | 7.20 | 82.50 | 1.20 | 91.00 | 54.50 | 86.71/96.94/98.74 | 2.52 | 0.50 |
| plain-frozen | 12.50 | 5.60 | 81.50 | 2.40 | 92.00 | 73.00 | 93.20/99.52/100.00 | 14.47 | 1.00 |

# LoRA-scale P2: 16-sample promotion (idx6 val-41)

Anchor adapter scaled in memory (`W=W_base+s*dW`); 41 validation behaviors x 16 samples, same fixed prompt/decoding/judge. Base and s=1 rows reuse the Q12 16-sample dev run in `humanjb-idx6-v1`; s=0.25/0.5 are new. Normal-dev refusal from the 100-example greedy pilot.

| point | ASR@1 % | ASR@4 % | ASR@16 % | normal-dev refusal % |
|---|---:|---:|---:|---:|
| base (s=0) | 1.83 | 2.44 | 2.44 | 1.00 |
| anchor s=0.25 | 2.59 | 3.05 | 4.88 | 0.00 |
| anchor s=0.5 | 6.55 | 11.30 | 14.63 | 0.00 |
| anchor s=1.0 | 34.91 | 43.16 | 48.78 | 0.00 |

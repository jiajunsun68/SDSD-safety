# Validated ablations results

Protocol: `sdsd-safety-standard200-v2`. HarmBench uses standard 200 direct requests and Wilson intervals. XSTest string-matched safe refusal is the primary refusal metric; semantic refusal is supplementary.

| Run | Seed | Steps | ASR % (95% CI) | Safe refusal % (string, primary) | Safe refusal % (semantic, supplementary) | GSM8K strict % | GSM8K flexible % |
|---|---:|---:|---:|---:|---:|---:|---:|
| sdsd_v2_external_k1_100_seed42 | 42 | 100 | 0.50 (0.09, 2.78) | 9.60 | 12.00 | 93.00 | 93.00 |
| sdsd_v2_k3_logpool_100_seed42 | 42 | 100 | 0.50 (0.09, 2.78) | 6.80 | 10.00 | 92.50 | 92.50 |
| sdsd_v2_anchor_k1_100_seed42 | 42 | 100 | 0.50 (0.09, 2.78) | 6.80 | 10.00 | 92.50 | 92.50 |
| sdsd_v2_k3_linpool_100_seed42 | 42 | 100 | 1.00 (0.27, 3.57) | 7.20 | 9.60 | 91.50 | 91.50 |

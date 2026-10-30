# Fixed-attack repeated-sampling ASR (dev)

Protocol `sdsd-safety-hb-humanjb-fixed5-dev-v1`; HarmBench standard-validation behaviors (n=41); official HumanJailbreaks prompt index 5 (source hash recorded); temperature 1, top-p 1, top-k 0, max 320 new tokens; classifier reads the original behavior only. This is a fixed prompt protocol, not a transfer or adaptive attack.

| Run | ASR@1 % | ASR@4 % | ASR@16 % | Unknown |
|---|---:|---:|---:|---:|
| base_v2 | 0.00 | 0.00 | 0.00 | 0 |
| opsa_v2_100_seed42 | 0.00 | 0.00 | 0.00 | 0 |
| sdsd_v2_k1_100_seed42 | 0.00 | 0.00 | 0.00 | 0 |
| sdsd_v2_anchor_k1_100_seed42 | 0.00 | 0.00 | 0.00 | 0 |
| sdsd_v2_external_k1_100_seed42 | 0.00 | 0.00 | 0.00 | 0 |

| A minus B | k | ΔASR (pp) | 95% CI | CI excludes 0 |
|---|---:|---:|---|---|
| opsa_v2_100_seed42 − base_v2 | 1 | +0.00 | [+0.00, +0.00] | False |
| opsa_v2_100_seed42 − base_v2 | 4 | +0.00 | [+0.00, +0.00] | False |
| opsa_v2_100_seed42 − base_v2 | 16 | +0.00 | [+0.00, +0.00] | False |
| sdsd_v2_k1_100_seed42 − base_v2 | 1 | +0.00 | [+0.00, +0.00] | False |
| sdsd_v2_k1_100_seed42 − base_v2 | 4 | +0.00 | [+0.00, +0.00] | False |
| sdsd_v2_k1_100_seed42 − base_v2 | 16 | +0.00 | [+0.00, +0.00] | False |
| sdsd_v2_anchor_k1_100_seed42 − base_v2 | 1 | +0.00 | [+0.00, +0.00] | False |
| sdsd_v2_anchor_k1_100_seed42 − base_v2 | 4 | +0.00 | [+0.00, +0.00] | False |
| sdsd_v2_anchor_k1_100_seed42 − base_v2 | 16 | +0.00 | [+0.00, +0.00] | False |
| sdsd_v2_external_k1_100_seed42 − base_v2 | 1 | +0.00 | [+0.00, +0.00] | False |
| sdsd_v2_external_k1_100_seed42 − base_v2 | 4 | +0.00 | [+0.00, +0.00] | False |
| sdsd_v2_external_k1_100_seed42 − base_v2 | 16 | +0.00 | [+0.00, +0.00] | False |

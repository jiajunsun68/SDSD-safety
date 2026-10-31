# Fixed-attack repeated-sampling ASR

Protocol `sdsd-safety-hb-humanjb-idx6-test159-v1`; HarmBench standard behaviors (n=159; source/hash in manifest); official HumanJailbreaks prompt index 6 (source hash recorded); temperature 1, top-p 1, top-k 0, max 320 new tokens; classifier reads the original behavior only. This is a fixed prompt protocol, not a transfer or adaptive attack.

| Run | ASR@1 % | ASR@4 % | ASR@16 % | Unknown |
|---|---:|---:|---:|---:|
| base_v2 | 0.94 | 1.50 | 2.52 | 0 |
| opsa_v2_100_seed42 | 2.75 | 5.29 | 10.69 | 0 |
| sdsd_v2_k1_100_seed42 | 7.90 | 12.04 | 15.09 | 0 |
| sdsd_v2_anchor_k1_100_seed42 | 37.89 | 49.91 | 57.23 | 0 |
| sdsd_v2_external_k1_100_seed42 | 20.83 | 28.90 | 32.70 | 0 |
| sdsd_v3_anchor_plainfrozen_100_seed42 | 1.26 | 2.31 | 3.14 | 0 |
| sdsd_v3_k1_refkl03_100_seed42 | 43.67 | 54.08 | 59.75 | 0 |

| A minus B | k | ΔASR (pp) | 95% CI | CI excludes 0 |
|---|---:|---:|---|---|
| opsa_v2_100_seed42 − base_v2 | 1 | +1.81 | [+0.67, +3.30] | True |
| opsa_v2_100_seed42 − base_v2 | 4 | +3.79 | [+1.90, +6.00] | True |
| opsa_v2_100_seed42 − base_v2 | 16 | +8.18 | [+4.40, +12.58] | True |
| sdsd_v2_k1_100_seed42 − base_v2 | 1 | +6.96 | [+3.97, +10.26] | True |
| sdsd_v2_k1_100_seed42 − base_v2 | 4 | +10.54 | [+6.39, +15.17] | True |
| sdsd_v2_k1_100_seed42 − base_v2 | 16 | +12.58 | [+7.55, +18.24] | True |
| sdsd_v2_anchor_k1_100_seed42 − base_v2 | 1 | +36.95 | [+30.74, +43.32] | True |
| sdsd_v2_anchor_k1_100_seed42 − base_v2 | 4 | +48.41 | [+41.12, +55.71] | True |
| sdsd_v2_anchor_k1_100_seed42 − base_v2 | 16 | +54.72 | [+47.17, +62.26] | True |
| sdsd_v2_external_k1_100_seed42 − base_v2 | 1 | +19.89 | [+14.70, +25.31] | True |
| sdsd_v2_external_k1_100_seed42 − base_v2 | 4 | +27.40 | [+20.93, +34.01] | True |
| sdsd_v2_external_k1_100_seed42 − base_v2 | 16 | +30.19 | [+23.27, +37.11] | True |
| sdsd_v3_anchor_plainfrozen_100_seed42 − base_v2 | 1 | +0.31 | [-0.04, +0.90] | False |
| sdsd_v3_anchor_plainfrozen_100_seed42 − base_v2 | 4 | +0.81 | [+0.00, +1.96] | False |
| sdsd_v3_anchor_plainfrozen_100_seed42 − base_v2 | 16 | +0.63 | [+0.00, +1.89] | False |
| sdsd_v3_k1_refkl03_100_seed42 − base_v2 | 1 | +42.73 | [+36.20, +49.49] | True |
| sdsd_v3_k1_refkl03_100_seed42 − base_v2 | 4 | +52.58 | [+45.06, +59.87] | True |
| sdsd_v3_k1_refkl03_100_seed42 − base_v2 | 16 | +57.23 | [+49.69, +64.78] | True |

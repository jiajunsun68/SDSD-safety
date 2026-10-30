# Fixed-attack repeated-sampling ASR (dev)

Protocol `sdsd-safety-hb-humanjb-idx6-dev-v1`; HarmBench standard-validation behaviors (n=41); official HumanJailbreaks prompt index 6 (source hash recorded); temperature 1, top-p 1, top-k 0, max 320 new tokens; classifier reads the original behavior only. This is a fixed prompt protocol, not a transfer or adaptive attack.

| Run | ASR@1 % | ASR@4 % | ASR@16 % | Unknown |
|---|---:|---:|---:|---:|
| base_v2 | 1.83 | 2.44 | 2.44 | 0 |
| opsa_v2_100_seed42 | 2.59 | 3.92 | 4.88 | 0 |
| sdsd_v2_k1_100_seed42 | 6.71 | 10.93 | 14.63 | 0 |
| sdsd_v2_anchor_k1_100_seed42 | 34.91 | 43.16 | 48.78 | 0 |
| sdsd_v2_external_k1_100_seed42 | 16.62 | 23.00 | 26.83 | 0 |

| A minus B | k | ΔASR (pp) | 95% CI | CI excludes 0 |
|---|---:|---:|---|---|
| opsa_v2_100_seed42 − base_v2 | 1 | +0.76 | [+0.00, +1.98] | False |
| opsa_v2_100_seed42 − base_v2 | 4 | +1.48 | [+0.00, +4.45] | False |
| opsa_v2_100_seed42 − base_v2 | 16 | +2.44 | [+0.00, +7.32] | False |
| sdsd_v2_k1_100_seed42 − base_v2 | 1 | +4.88 | [+0.91, +10.06] | True |
| sdsd_v2_k1_100_seed42 − base_v2 | 4 | +8.49 | [+1.71, +17.05] | True |
| sdsd_v2_k1_100_seed42 − base_v2 | 16 | +12.20 | [+2.44, +21.95] | True |
| sdsd_v2_anchor_k1_100_seed42 − base_v2 | 1 | +33.08 | [+20.73, +46.19] | True |
| sdsd_v2_anchor_k1_100_seed42 − base_v2 | 4 | +40.72 | [+26.75, +55.22] | True |
| sdsd_v2_anchor_k1_100_seed42 − base_v2 | 16 | +46.34 | [+31.71, +60.98] | True |
| sdsd_v2_external_k1_100_seed42 − base_v2 | 1 | +14.79 | [+6.40, +24.70] | True |
| sdsd_v2_external_k1_100_seed42 − base_v2 | 4 | +20.57 | [+9.43, +32.73] | True |
| sdsd_v2_external_k1_100_seed42 − base_v2 | 16 | +24.39 | [+12.20, +39.02] | True |

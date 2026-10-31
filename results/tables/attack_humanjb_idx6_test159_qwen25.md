# Fixed-attack repeated-sampling ASR

Protocol `sdsd-safety-hb-humanjb-idx6-test159-qwen25-v1`; HarmBench standard behaviors (n=159; source/hash in manifest); official HumanJailbreaks prompt index 6 (source hash recorded); temperature 1, top-p 1, top-k 0, max 320 new tokens; classifier reads the original behavior only. This is a fixed prompt protocol, not a transfer or adaptive attack.

| Run | ASR@1 % | ASR@4 % | ASR@16 % | Unknown |
|---|---:|---:|---:|---:|
| base_qwen25_v1 | 93.12 | 99.87 | 100.00 | 0 |
| sdsd_v3_qwen25_anchor_100_seed42 | 86.71 | 96.94 | 98.74 | 0 |
| sdsd_v3_qwen25_plainfrozen_100_seed42 | 93.20 | 99.52 | 100.00 | 0 |

| A minus B | k | ΔASR (pp) | 95% CI | CI excludes 0 |
|---|---:|---:|---|---|
| sdsd_v3_qwen25_anchor_100_seed42 − base_qwen25_v1 | 1 | -6.41 | [-9.59, -3.34] | True |
| sdsd_v3_qwen25_anchor_100_seed42 − base_qwen25_v1 | 4 | -2.93 | [-5.25, -1.10] | True |
| sdsd_v3_qwen25_anchor_100_seed42 − base_qwen25_v1 | 16 | -1.26 | [-3.14, +0.00] | False |
| sdsd_v3_qwen25_plainfrozen_100_seed42 − base_qwen25_v1 | 1 | +0.08 | [-1.34, +1.45] | False |
| sdsd_v3_qwen25_plainfrozen_100_seed42 − base_qwen25_v1 | 4 | -0.36 | [-0.89, -0.00] | True |
| sdsd_v3_qwen25_plainfrozen_100_seed42 − base_qwen25_v1 | 16 | +0.00 | [+0.00, +0.00] | False |

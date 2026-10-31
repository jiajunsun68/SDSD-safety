# Fixed-attack repeated-sampling ASR

Protocol `sdsd-safety-hb-humanjb-idx8-test159-indep-v1`; HarmBench standard behaviors (n=159; source/hash in manifest); official HumanJailbreaks prompt index 8 (source hash recorded); temperature 1, top-p 1, top-k 0, max 320 new tokens; classifier reads the original behavior only. This is a fixed prompt protocol, not a transfer or adaptive attack.

| Run | ASR@1 % | ASR@4 % | ASR@16 % | Unknown |
|---|---:|---:|---:|---:|
| base_v2 | 0.00 | 0.00 | 0.00 | 0 |
| sdsd_v2_anchor_k1_100_seed42 | 0.00 | 0.00 | 0.00 | 0 |
| sdsd_v3_anchor_plainfrozen_100_seed42 | 0.00 | 0.00 | 0.00 | 0 |

| A minus B | k | ΔASR (pp) | 95% CI | CI excludes 0 |
|---|---:|---:|---|---|
| sdsd_v2_anchor_k1_100_seed42 − base_v2 | 1 | +0.00 | [+0.00, +0.00] | False |
| sdsd_v2_anchor_k1_100_seed42 − base_v2 | 4 | +0.00 | [+0.00, +0.00] | False |
| sdsd_v2_anchor_k1_100_seed42 − base_v2 | 16 | +0.00 | [+0.00, +0.00] | False |
| sdsd_v3_anchor_plainfrozen_100_seed42 − base_v2 | 1 | +0.00 | [+0.00, +0.00] | False |
| sdsd_v3_anchor_plainfrozen_100_seed42 − base_v2 | 4 | +0.00 | [+0.00, +0.00] | False |
| sdsd_v3_anchor_plainfrozen_100_seed42 − base_v2 | 16 | +0.00 | [+0.00, +0.00] | False |

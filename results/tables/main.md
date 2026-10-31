# Validated main results

Protocol: `sdsd-safety-standard200-v2`. HarmBench uses standard 200 direct requests and Wilson intervals. XSTest string-matched safe refusal is the primary refusal metric; semantic refusal is supplementary.

| Run | Seed | Steps | ASR greedy % (95% CI) | Safe refusal % (string, primary) | Safe refusal % (semantic, supplementary) | GSM8K lm-eval flexible % (primary) | GSM8K exact % (diagnostic) | GSM8K strict % (diagnostic) | GSM8K flexible % (diagnostic) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_v2 | 42 | 0 | 0.50 (0.09, 2.78) | 5.60 | 7.60 | 91.00 | 43.00 | 86.50 | 93.50 |
| sft_v2_100_seed42 | 42 | 100 | 3.00 (1.38, 6.39) | 19.60 | 20.00 | 90.50 | 0.00 | 72.00 | 72.00 |
| grpo_v2_100_seed42 | 42 | 100 | 0.50 (0.09, 2.78) | 27.60 | 31.60 | 91.50 | 42.50 | 67.50 | 69.50 |
| opsa_v2_100_seed42 | 42 | 100 | 0.50 (0.09, 2.78) | 14.40 | 16.40 | 91.00 | 17.50 | 90.00 | 92.00 |
| sdsd_v2_k1_100_seed42 | 42 | 100 | 0.50 (0.09, 2.78) | 9.20 | 10.00 | 88.50 | 3.00 | 91.00 | 91.00 |

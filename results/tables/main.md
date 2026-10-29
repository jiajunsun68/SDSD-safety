| Method | Steps | HarmBench ASR↓ | XSTest safe refusal↓ | XSTest unsafe refusal↑ | GSM8K↑ |
|---|---|---|---|---|---|
| Base | 0 | 4.75 [2.75, 7.00] | 7.6 | 82.0 | 90.5 |
| Safety SFT (refusal targets) | 200 | 2.00 [0.75, 3.50] | 92.8 | 100.0 | 87.0 |
| Safety GRPO (margin + refusal) | 200 | 0.00 [0.00, 0.00] | 98.0 | 100.0 | 80.5 |
| OPSA | 200 | 5.25 [3.25, 7.50] | 35.2 | 95.0 | 91.5 |
| SDSD-Safety (K=1, self refusal) | 200 | 0.75 [0.00, 1.75] | 73.2 | 100.0 | 70.5 |

XSTest columns are the Qwen3Guard semantic refusal rates. Official XSTest string-match rates are in `main.csv`; the official matcher misses "You cannot ..." style refusals (e.g. OPSA reports 33.6% safe / 95.5% unsafe officially).

Ablations:

| Variant | Steps | HarmBench ASR↓ | XSTest safe↓ | XSTest unsafe↑ | GSM8K↑ |
|---|---|---|---|---|---|
| SDSD, external refusal demos K=1 | 100 | 5.25 | 17.6 | 94.0 | 87.5 |
| SDSD, self refusal demos K=3 | 100 | 12.00 | 25.6 | 98.5 | 90.0 |
| SFT on external safe answers (exploratory) | 200 | 28.50 | 6.8 | 59.0 | 71.0 |
| SDSD, any-safe demos | 200 | invalid (repetition collapse) | — | — | — |

The two SDSD ablations use 100 steps for budget; the main SDSD run uses 200.

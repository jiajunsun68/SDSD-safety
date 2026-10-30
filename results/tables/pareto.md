# Standard-metric Pareto analysis

All numbers are standard benchmark metrics; no composite score is used.
Dominance uses primary metrics only: ASR lower, official XSTest safe refusal lower, GSM8K strict higher. Semantic refusal is explanatory.

| run | ASR | XSTest safe (official) | XSTest unsafe (official) | XSTest safe (semantic) | XSTest unsafe (semantic) | GSM8K strict | ΔASR vs base | Δsafe vs base | ΔGSM8K vs base | dominated by |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base_v2 | 0.0050 | 0.056 | 0.775 | 0.076 | 0.820 | 0.865 | +0.0000 | +0.000 | +0.000 | — |
| grpo_v2_100_seed42 | 0.0050 | 0.276 | 0.990 | 0.316 | 0.965 | 0.675 | +0.0000 | +0.220 | -0.190 | base_v2, opsa_v2_100_seed42, sdsd_v2_anchor_k1_100_seed42, sdsd_v2_external_k1_100_seed42, sdsd_v2_k1_100_seed42, sdsd_v2_k3_logpool_100_seed42 |
| opsa_v2_100_seed42 | 0.0050 | 0.144 | 0.895 | 0.164 | 0.875 | 0.900 | +0.0000 | +0.088 | +0.035 | sdsd_v2_anchor_k1_100_seed42, sdsd_v2_external_k1_100_seed42, sdsd_v2_k1_100_seed42, sdsd_v2_k3_logpool_100_seed42 |
| sdsd_v2_anchor_k1_100_seed42 | 0.0050 | 0.068 | 0.795 | 0.100 | 0.830 | 0.925 | +0.0000 | +0.012 | +0.060 | — |
| sdsd_v2_external_k1_100_seed42 | 0.0050 | 0.096 | 0.860 | 0.120 | 0.860 | 0.930 | +0.0000 | +0.040 | +0.065 | — |
| sdsd_v2_k1_100_seed42 | 0.0050 | 0.092 | 0.825 | 0.100 | 0.830 | 0.910 | +0.0000 | +0.036 | +0.045 | sdsd_v2_anchor_k1_100_seed42, sdsd_v2_k3_logpool_100_seed42 |
| sdsd_v2_k3_linpool_100_seed42 | 0.0100 | 0.072 | 0.755 | 0.096 | 0.830 | 0.915 | +0.0050 | +0.016 | +0.050 | sdsd_v2_anchor_k1_100_seed42, sdsd_v2_k3_logpool_100_seed42 |
| sdsd_v2_k3_logpool_100_seed42 | 0.0050 | 0.068 | 0.795 | 0.100 | 0.835 | 0.925 | +0.0000 | +0.012 | +0.060 | — |
| sft_v2_100_seed42 | 0.0300 | 0.196 | 0.960 | 0.200 | 0.945 | 0.720 | +0.0250 | +0.140 | -0.145 | base_v2, opsa_v2_100_seed42, sdsd_v2_anchor_k1_100_seed42, sdsd_v2_external_k1_100_seed42, sdsd_v2_k1_100_seed42, sdsd_v2_k3_linpool_100_seed42, sdsd_v2_k3_logpool_100_seed42 |

## Pairwise research comparisons

- **K=3 mean-KL vs probability-mean**: `sdsd_v2_k3_logpool_100_seed42` ASR 0.0050, safe 0.068, GSM8K 0.925; `sdsd_v2_k3_linpool_100_seed42` ASR 0.0100, safe 0.072, GSM8K 0.915 → **A dominates B**.
- **external-demo K1 vs EMA+frozen(0.25)**: `sdsd_v2_external_k1_100_seed42` ASR 0.0050, safe 0.096, GSM8K 0.930; `sdsd_v2_anchor_k1_100_seed42` ASR 0.0050, safe 0.068, GSM8K 0.925 → **neither dominates**.
- **self-demo K1 vs EMA+frozen(0.25)**: `sdsd_v2_k1_100_seed42` ASR 0.0050, safe 0.092, GSM8K 0.910; `sdsd_v2_anchor_k1_100_seed42` ASR 0.0050, safe 0.068, GSM8K 0.925 → **B dominates A**.
- **self-demo K1 vs external-demo K1**: `sdsd_v2_k1_100_seed42` ASR 0.0050, safe 0.092, GSM8K 0.910; `sdsd_v2_external_k1_100_seed42` ASR 0.0050, safe 0.096, GSM8K 0.930 → **neither dominates**.
- **SDSD K1 vs OPSA**: `sdsd_v2_k1_100_seed42` ASR 0.0050, safe 0.092, GSM8K 0.910; `opsa_v2_100_seed42` ASR 0.0050, safe 0.144, GSM8K 0.900 → **A dominates B**.
- **SDSD K1 vs SFT**: `sdsd_v2_k1_100_seed42` ASR 0.0050, safe 0.092, GSM8K 0.910; `sft_v2_100_seed42` ASR 0.0300, safe 0.196, GSM8K 0.720 → **A dominates B**.
- **SDSD K1 vs GRPO**: `sdsd_v2_k1_100_seed42` ASR 0.0050, safe 0.092, GSM8K 0.910; `grpo_v2_100_seed42` ASR 0.0050, safe 0.276, GSM8K 0.675 → **A dominates B**.

## Reading

The base model has only one harmful generation out of 200 standard behaviors; ASR is at a near-floor and cannot fall by more than 0.5 points. Interpret safety changes together with XSTest unsafe refusal and the capability columns, not ASR alone.

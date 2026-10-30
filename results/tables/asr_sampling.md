# Repeated-sampling ASR at fixed decoding

Protocol: `sdsd-safety-asr-sample-v3`. Each behavior contributes the same 16 independent samples at temperature 1, top-p 1, top-k 0, max 192 new tokens. ASR@k is the combinatorial probability that k of those fixed samples contain at least one harmful response, averaged over requests. This is not an adaptive attack budget. CIs are paired request-level percentile bootstraps; unknown judgments are bounded, not imputed.

| Run | ASR@1 % (95% CI) | ASR@4 % (95% CI) | ASR@16 % (95% CI) | Unknown labels |
|---|---:|---:|---:|---:|
| — | — | — | — | no validated repeated-sampling results yet |

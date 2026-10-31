# Teacher x context x prefix probe

Fixed continuations from existing anchor-attack/normal-dev responses (40 tokens); plain request vs safe demonstration; no prefix vs first 8 tokens as compliant prefix. Values are teacher-forced mean token log-probabilities; KL is first-scored-token KL(EMA||Base) on 8 requests.

| teacher | context/prefix | harmful logP | normal logP | KL harmful | KL normal |
|---|---|---:|---:|---:|---:|
| base | plain_none | -4.8095 | -2.4775 | 0.0000 | 0.0000 |
| base | plain_short | -2.5512 | -2.1859 | 0.0000 | 0.0000 |
| base | demo_none | -4.8246 | -2.2853 | 0.0000 | 0.0000 |
| base | demo_short | -2.2348 | -2.1368 | 0.0000 | 0.0000 |
| ema_k1 | plain_none | -5.2519 | -2.3692 | 0.0028 | 8.7047 |
| ema_k1 | plain_short | -2.4908 | -2.2706 | 0.1300 | 0.0920 |
| ema_k1 | demo_none | -5.1881 | -2.3376 | 0.0001 | 1.4225 |
| ema_k1 | demo_short | -2.2019 | -2.2483 | 0.1634 | 0.0305 |
| ema_anchor | plain_none | -5.2160 | -2.3535 | 0.0028 | 9.2710 |
| ema_anchor | plain_short | -2.5466 | -2.2837 | 0.0526 | 0.0159 |
| ema_anchor | demo_none | -5.1556 | -2.3156 | 0.0001 | 1.5514 |
| ema_anchor | demo_short | -2.2356 | -2.2420 | 0.1018 | 0.0019 |

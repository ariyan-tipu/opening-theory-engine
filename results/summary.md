# Evaluation summary (colab-t4, 150 held-out positions)

| Metric | baseline | sft | dpo |
| --- | --- | --- | --- |
| format_rate | 0.0133 | 1.0 | 1.0 |
| verdict_exact | 0.2933 | 0.5 | 0.5 |
| verdict_within_one | 0.9467 | 0.7933 | 0.7933 |
| eval_within_one | 1.0 | 0.9 | 0.8933 |
| line_illegal_rate | 0.6667 | 0.9533 | 0.96 |
| claim_precision | 0.4702 | 0.5721 | 0.5745 |
| hallucination_rate | 1.48 | 1.76 | 1.7133 |
| verified_pass_rate | 0.0 | 0.02 | 0.02 |
| composite_score_mean | -2.8913 | -1.3601 | -1.3211 |
| fluency: mean_tokens | 75.91 | 34.71 | 36.92 |
| fluency: distinct_2 | 0.9781 | 0.995 | 0.9996 |
| fluency: repetition_rate | 0.0933 | 0.0 | 0.0 |

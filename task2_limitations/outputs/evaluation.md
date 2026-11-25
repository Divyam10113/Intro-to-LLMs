# Evaluation

Reference: 3 limitations of MVFA stated in held-out citing papers (never indexed for retrieval).
Judge models: nvidia/nemotron-3-ultra-550b-a55b, z-ai/glm-5.3-flash

| system | #lims | C_GT (recall) | C_LLM | F1 | Faith. | Sound. | Import. |
|---|---|---|---|---|---|---|---|
| full | 27 | 66.7% | 7.4% | 0.133 | 5.00 | 5.00 | 3.74 |
| full_nocite | 25 | 33.3% | 4.0% | 0.071 | 5.00 | 5.00 | 3.72 |
| zero_shot | 15 | 33.3% | 6.7% | 0.111 | 5.00 | 5.00 | 4.60 |
| old_pipeline | 11 | 66.7% | 27.3% | 0.387 | 5.00 | 5.00 | 4.00 |
| full_nim | 39 | 100.0% | 10.3% | 0.186 | 5.00 | 4.67 | 4.62 |

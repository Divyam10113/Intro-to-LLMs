# In-mask diagnostic: share of top-k patches inside the true mask (masked sources' valid/ abnormal images, selected checkpoint, seeds 3–5)

| arm | Brain fold (sources Liver, RESC) | Liver fold (sources Brain, RESC) | RESC+OCT17 fold (sources Brain, Liver) | mean of folds | analysis.md pooled (all rows) |
|---|---|---|---|---|---|
| MVFA (re-run) | 0.929 | 0.843 | 0.590 | 0.787 | 0.766 |
| MIL | 0.546 | 0.647 | 0.163 | 0.452 | 0.417 |
| PU+100% | 0.929 | 0.837 | 0.598 | 0.788 | 0.741 |
| MIL+100% | 0.919 | 0.829 | 0.589 | 0.779 | 0.732 |
| MIL+10% | 0.820 | 0.759 | 0.404 | 0.661 | 0.597 |
| MIL+25% | 0.865 | 0.788 | 0.481 | 0.711 | 0.654 |

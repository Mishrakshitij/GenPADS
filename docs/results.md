# Results reported in the paper

All values on this page are published results from [Mishra, Firdaus, and Ekbal (2023)](https://doi.org/10.1371/journal.pone.0278323), Tables 4–6. They are reference measurements, not results obtained by running this code release. The complete values are available in [paper-results.csv](paper-results.csv); blank CSV fields mean that a metric does not apply or was not reported.

Table 4 reports PC F1 and generation-module G perplexity (PP), BLEU, NIST, METEOR (MET), and ROUGE-2 F1 (R-2 F1). The paper does not specify the averaging convention for its PC F1; the implementation reports macro-F1.

| Domain | PC F1 | G PP | G BLEU | G NIST | G MET | G R-2 F1 |
|---|---:|---:|---:|---:|---:|---:|
| flights | 0.92 | 1.912 | 0.052 | 0.186 | 0.641 | 0.472 |
| food-ordering | 0.96 | 1.698 | 0.050 | 0.214 | 0.758 | 0.452 |
| hotels | 0.94 | 1.972 | 0.065 | 0.207 | 0.664 | 0.504 |
| movies | 0.95 | 2.137 | 0.039 | 0.1618 | 0.654 | 0.469 |
| music | 0.93 | 2.367 | 0.037 | 0.133 | 0.555 | 0.379 |
| restaurant-search | 0.95 | 2.156 | 0.047 | 0.162 | 0.669 | 0.494 |
| sports | 0.92 | 1.762 | 0.018 | 0.069 | 0.739 | 0.585 |

The following summaries show Table 5 success rate (SR), dialogue length (DL), and politeness (POL). SR and POL are fractions on a 0–1 scale; DL is measured in turns. BL is the baseline reward and PRRP is the politeness reward with repetition penalty. The CSV also includes the dialogue systems' METEOR and ROUGE-2 F1 values.

**GenPADS**

| Domain | BL SR | PRRP SR | BL DL | PRRP DL | BL POL | PRRP POL |
|---|---:|---:|---:|---:|---:|---:|
| flights | 0.67 | 0.79 | 10.8 | 10.7 | 0.587 | 0.851 |
| food-ordering | 0.69 | 0.86 | 13.8 | 11.5 | 0.656 | 0.936 |
| hotels | 0.74 | 0.82 | 10.9 | 9.9 | 0.849 | 0.893 |
| movies | 0.77 | 0.84 | 9.9 | 9.5 | 0.744 | 0.888 |
| music | 0.71 | 0.86 | 9.7 | 9.4 | 0.910 | 0.959 |
| restaurant-search | 0.79 | 0.82 | 9.5 | 8.5 | 0.418 | 0.920 |
| sports | 0.64 | 0.81 | 11.3 | 10.9 | 0.806 | 0.948 |

**RetrievalPADS**

| Domain | BL SR | PRRP SR | BL DL | PRRP DL | BL POL | PRRP POL |
|---|---:|---:|---:|---:|---:|---:|
| flights | 0.67 | 0.77 | 11.1 | 10.3 | 0.674 | 0.842 |
| food-ordering | 0.686 | 0.84 | 12.9 | 12.6 | 0.597 | 0.908 |
| hotels | 0.71 | 0.82 | 12.8 | 10.3 | 0.804 | 0.864 |
| movies | 0.74 | 0.83 | 11.8 | 9.7 | 0.694 | 0.865 |
| music | 0.71 | 0.84 | 9.6 | 9.4 | 0.881 | 0.921 |
| restaurant-search | 0.75 | 0.78 | 11.7 | 9.7 | 0.381 | 0.940 |
| sports | 0.64 | 0.79 | 13.6 | 11.5 | 0.795 | 0.915 |

Table 5 also reports the supervised dialogue generator baseline (DG). DG has no dialogue manager, so that table does not assign it task success, dialogue length, or politeness scores.

| Domain | PP | BLEU | NIST | MET | R-2 F1 |
|---|---:|---:|---:|---:|---:|
| flights | 6.18 | 0.038 | 0.132 | 0.127 | 0.059 |
| food-ordering | 3.05 | 0.027 | 0.172 | 0.387 | 0.345 |
| hotels | 7.15 | 0.087 | 0.261 | 0.146 | 0.078 |
| movies | 7.45 | 0.015 | 0.058 | 0.146 | 0.086 |
| music | 11.4 | 0.007 | 0.33 | 0.231 | 0.156 |
| restaurant-search | 8.46 | 0.046 | 0.153 | 0.165 | 0.089 |
| sports | 5.14 | 0.007 | 0.032 | 0.270 | 0.163 |

Table 6 gives human evaluation averages across domains on a five-point scale.

| Reward | Fluency | Informativeness | Politeness adaptability | Diversity |
|---|---:|---:|---:|---:|
| BL | 3.87 | 3.32 | 3.18 | 3.54 |
| PRRP | 4.16 | 3.87 | 4.08 | 3.91 |

Release validation covers reward branches, masks, discounted returns, policy gradients with exploration, stochastic simulator rates, frozen evaluation, checkpoint round trips, and isolated generator random streams. Tiny local DistilBERT and BART models complete training, save/reload, and evaluation checks. These checks establish executable behavior; they do not reproduce the full supervised training, 8,000-episode policy experiments, or human evaluation behind the paper tables. See [method.md](method.md) for implementation choices and comparison limits.

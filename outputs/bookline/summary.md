# Sportsbook-line veto: historical test on 2025-26 fills

151 games sampled (seed 20261016), 151 with book lines in the feed; book quote at the exact line for 66% of fills.

| Fills | Count | Games | Cents per contract | t | Over hit rate |
|---|---:|---:|---:|---:|---:|
| 1. all fills in the sample | 4,862 | 151 | +2.04 | 1.8 | 0.431 |
| 2. fills with a book quote at the exact line | 3,197 | 150 | +2.15 | 1.58 | 0.458 |
| 2b. fills with no book quote | 1,665 | 134 | +1.83 | 1.38 | 0.379 |
| 3. book veto: book edge >= 0 | 1,647 | 146 | +3.81 | 2.39 | 0.520 |
| 3b. removed by the book veto | 1,550 | 132 | +0.38 | 0.25 | 0.392 |
| 4. book veto at a 2 cent margin | 476 | 123 | +2.23 | 0.98 | 0.613 |
| 5. model veto on covered fills | 2,074 | 139 | +2.69 | 1.6 | 0.468 |
| 6. both vetoes | 1,149 | 133 | +3.74 | 1.91 | 0.534 |

Kept minus removed by the book veto: +3.44¢ per contract, clustered t 2.42.
Veto against all covered fills: +1.66¢.
**Pre-registered pass (≥ 1¢ gain, t ≥ 2, ≥ 2,000 covered fills): True.**

Book probability against what happened (covered fills):

| Book p(over) | Fills | Over hit | Mean book p | Mean Kalshi yes ask |
|---|---:|---:|---:|---:|
| (0.1, 0.2] | 232 | 0.151 | 0.160 | 0.158 |
| (0.2, 0.3] | 279 | 0.197 | 0.248 | 0.251 |
| (0.3, 0.4] | 484 | 0.366 | 0.351 | 0.358 |
| (0.4, 0.5] | 747 | 0.418 | 0.452 | 0.463 |
| (0.5, 0.6] | 683 | 0.530 | 0.547 | 0.569 |
| (0.6, 0.7] | 422 | 0.600 | 0.649 | 0.670 |
| (0.7, 0.8] | 257 | 0.778 | 0.745 | 0.772 |
| (0.8, 0.9] | 93 | 0.763 | 0.819 | 0.844 |

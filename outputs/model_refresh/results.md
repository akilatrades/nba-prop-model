# Model refresh study, 2025-26 holdout

535,731 player-game-lines; 53,549 resting orders re-priced.

| Variant | Log loss | vs control | t | Brier |
|---|---:|---:|---:|---:|
| control | 0.30380 | +0.00000 | None | 0.09648 |
| gbm | 0.30247 | -0.00134 | -5.81 | 0.09606 |
| hier | 0.30413 | +0.00033 | 3.05 | 0.09661 |
| iso | 0.30387 | +0.00007 | 1.78 | 0.09650 |

| Variant | Placed | Fills | Profit per 100 quoted orders (¢) | t | ¢/fill | Diff vs control | t |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 35,404 | 20,641 | +112.9 | 6.07 | 2.93 | +0.0 | None |
| gbm | 34,303 | 19,787 | +107.9 | 5.83 | 2.92 | -5.0 | -0.58 |
| hier | 33,602 | 19,459 | +122.4 | 6.84 | 3.37 | +9.5 | 1.62 |
| iso | 35,224 | 20,560 | +113.2 | 6.09 | 2.95 | +0.3 | 0.12 |

| Period | Ordering | End ($500 start) | Max DD | Orders | Fills | Profit per 100 orders (¢) |
|---|---|---:|---:|---:|---:|---:|
| whole season | edge-first | 1,849 | 44.2% | 6,190 | 2,107 | +21.79 |
| whole season | edge / sd | 1,569 | 45.8% | 6,319 | 2,190 | +16.92 |
| whole season | random | 3,222 | 26.3% | 5,724 | 3,033 | +47.56 |
| first half | edge-first | 907 | 27.7% | 3,060 | 1,067 | +13.30 |
| first half | edge / sd | 1,130 | 19.8% | 3,079 | 1,086 | +20.45 |
| first half | random | 1,660 | 17.6% | 2,946 | 1,399 | +39.37 |
| second half | edge-first | 1,161 | 21.4% | 2,750 | 863 | +24.05 |
| second half | edge / sd | 783 | 33.7% | 2,655 | 846 | +10.67 |
| second half | random | 1,210 | 30.2% | 2,644 | 1,556 | +26.86 |

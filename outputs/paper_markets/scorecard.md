# Model-free "no" seller paper tests (other markets)

One $500 paper bankroll per series; rule in kalshi/paper_markets.py. Series marked +model use the NFL model as veto (markets/nfl/paper.py).

| Series | Orders | Open | Fill rate | Fills settled | ¢/contract | t | Bankroll |
|---|---:|---:|---:|---:|---:|---:|---:|
| KXITFMATCH | 65 | 3 | 84% | 52 | -2.59 | -1.22 | $423.53 |
| KXNASCARRACE | 17 | 17 |  | 0 |  |  | $500.00 |
| KXLIGAMXSPREAD | 12 | 12 |  | 0 |  |  | $500.00 |
| KXLALIGASPREAD | 4 | 0 | 50% | 2 | 15.73 | 0.0 | $501.80 |
| KXNFLANYTD | 0 | 0 |  | 0 |  |  | $500.00 |
| KXNFLRECYDS | 0 | 0 |  | 0 |  |  | $500.00 |
| KXNFLANYTD+model | 0 | 0 |  | 0 |  |  | $500.00 |
| KXNFLRECYDS+model | 0 | 0 |  | 0 |  |  | $500.00 |

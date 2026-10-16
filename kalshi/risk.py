"""Risk limits for live trading. Every order passes through ``check`` first.

Limits (dollars unless noted), read from ``RiskLimits``:

- ``max_order``: cost of one order (contracts x price).
- ``max_night``: total cost of orders resting or filled for one game day.
- ``max_open``: total cost of everything resting or filled, all days.
- ``max_per_game``: total cost in one game (one blowout should not hit more
  than this).
- ``daily_loss_stop``: once the day's realized loss reaches this, no new
  orders until the next day.
- ``kill_file``: if this file exists, nothing is placed and everything
  resting is cancelled. A manual off switch.
- ``max_orders_per_run``: a sanity cap; a bug that generates 5,000 orders
  stops here.
- ``max_orders_per_game``: how many separate lines we will sit on in one
  game. Every "no" order in a game wins on the same quiet night and loses on
  the same shootout, so this caps the correlated exposure.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RiskLimits:
    max_order: float = 5.0
    max_night: float = 75.0
    max_open: float = 150.0
    max_per_game: float = 25.0
    daily_loss_stop: float = 25.0
    max_orders_per_run: int = 150
    max_orders_per_game: int = 6
    kill_file: Path = Path("outputs/live/KILL")


@dataclass
class Exposure:
    """What is already committed, from the ledger: dollars by day and game."""

    by_day: dict
    by_game: dict
    open_total: float
    realized_today: float
    orders_by_game: dict = None

    def __post_init__(self):
        if self.orders_by_game is None:
            self.orders_by_game = {}


class Rejected(Exception):
    pass


def check(order_cost, day, game, limits, exposure, placed_this_run):
    """Raise ``Rejected`` with the reason, or return None if the order may go."""
    if limits.kill_file.exists():
        raise Rejected("kill file present")
    if placed_this_run >= limits.max_orders_per_run:
        raise Rejected(f"more than {limits.max_orders_per_run} orders in one run")
    if order_cost > limits.max_order + 1e-9:
        raise Rejected(
            f"order ${order_cost:.2f} over the ${limits.max_order:.2f} limit"
        )
    if exposure.realized_today <= -limits.daily_loss_stop:
        raise Rejected(f"daily loss stop hit (${-exposure.realized_today:.2f})")
    if exposure.by_day.get(day, 0.0) + order_cost > limits.max_night + 1e-9:
        raise Rejected(f"night cap ${limits.max_night:.2f} reached for {day}")
    if exposure.by_game.get(game, 0.0) + order_cost > limits.max_per_game + 1e-9:
        raise Rejected(f"game cap ${limits.max_per_game:.2f} reached for {game}")
    if exposure.open_total + order_cost > limits.max_open + 1e-9:
        raise Rejected(f"open exposure cap ${limits.max_open:.2f} reached")
    if exposure.orders_by_game.get(game, 0) >= limits.max_orders_per_game:
        raise Rejected(f"{limits.max_orders_per_game} orders already in game {game}")


def commit(exposure, order_cost, day, game):
    exposure.by_day[day] = exposure.by_day.get(day, 0.0) + order_cost
    exposure.by_game[game] = exposure.by_game.get(game, 0.0) + order_cost
    exposure.open_total += order_cost
    exposure.orders_by_game[game] = exposure.orders_by_game.get(game, 0) + 1

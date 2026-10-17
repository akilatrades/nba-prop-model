"""Adversarial checks for the non-trading forward-validation scorecard."""

import pandas as pd

from live.forward_validation import scorecard, write_outputs


NOW = pd.Timestamp("2026-11-20T20:00:00Z")


def _report():
    return pd.DataFrame(
        [
            {
                "order_id": "a",
                "ticker": "AAA",
                "game_id": 1,
                "day": "2026-10-20",
                "accept_utc": "2026-10-20T19:00:00Z",
                "placed_utc": "2026-10-20T18:59:59Z",
                "expiration_utc": "2026-10-21T01:00:00Z",
                "filled": 4,
                "settled_contracts": 4,
                "settled_profit": 0.16,
                "qualifying_volume": 5,
                "qualifying_complete": True,
                "queue_ahead_at_start": 0,
            },
            {
                "order_id": "b",
                "ticker": "BBB",
                "game_id": 2,
                "day": "2026-10-22",
                "accept_utc": "2026-10-22T19:00:00Z",
                "expiration_utc": "2026-10-23T01:00:00Z",
                "filled": 1,
                "settled_contracts": 1,
                "settled_profit": -0.02,
                "qualifying_volume": 20,
                "qualifying_complete": True,
                "queue_ahead_at_start": 105,
            },
        ]
    )


def _fills():
    return pd.DataFrame(
        [
            {"order_id": "a", "ticker": "AAA", "action": "buy",
             "side": "no", "fill_id": "1"},
            {"order_id": "a", "ticker": "AAA", "action": "buy",
             "side": "no", "fill_id": "1"},  # deduplicated
            {"order_id": "b", "ticker": "BBB", "action": "buy",
             "side": "no", "fill_id": "2"},
        ]
    )


def test_weighted_capture_is_not_average_of_order_percentages():
    result = scorecard(_report(), _fills(), NOW)
    assert result["capture_volume_weighted"] == 0.2  # (4+1)/(5+20)
    assert result["settled_buy_fill_events"] == 2
    assert result["settled_contracts"] == 5
    assert result["cents_per_settled_contract"] == 2.8
    assert result["queue_fill_probability"]["100+"]["orders"] == 1
    assert result["status"] == "collecting"


def test_incomplete_or_impossible_volume_is_not_counted_as_good_capture():
    r = _report()
    r.loc[1, "qualifying_complete"] = False
    score = scorecard(r, _fills(), NOW)
    assert score["capture_volume_weighted"] == 0.8
    assert score["mature_volume_coverage"] == 0.5
    r.loc[1, "qualifying_complete"] = True
    r.loc[1, "filled"] = 50
    score = scorecard(r, _fills(), NOW)
    assert score["impossible_capture_rows"] == 1
    assert score["capture_volume_weighted"] == 0.8


def test_exit_trades_are_not_misrepresented_as_hold_to_settlement_profit():
    f = _fills()
    f.loc[len(f)] = {
        "order_id": "exit", "ticker": "AAA", "action": "sell",
        "side": "no", "fill_id": "3"
    }
    result = scorecard(_report(), f, NOW)
    assert result["exited_market_orders_excluded"] == 1
    assert result["settled_contracts"] == 1
    assert result["settled_buy_fill_events"] == 1


def test_no_live_orders_writes_valid_empty_outputs(tmp_path):
    result = write_outputs(tmp_path, pd.DataFrame(), pd.DataFrame(), NOW)
    assert result["status"] == "awaiting_live_orders"
    assert (tmp_path / "forward_validation.json").exists()
    assert (tmp_path / "forward_validation.md").exists()


def test_checkpoint_never_auto_changes_rules_or_money():
    r = pd.concat([_report().iloc[:1].assign(
        order_id=f"id-{i}", ticker=f"T{i}", game_id=i
    ) for i in range(500)], ignore_index=True)
    f = pd.DataFrame([
        {"order_id": f"id-{i}", "ticker": f"T{i}",
         "action": "buy", "side": "no", "fill_id": f"fill-{i}"}
        for i in range(500)
    ])
    result = scorecard(r, f, NOW)
    assert result["status"] == "checkpoint_ready_for_human_review"
    assert result["automatic_trade_changes"] is False
    assert result["settled_buy_fill_events"] == 500

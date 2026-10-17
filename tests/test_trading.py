import base64
import time

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from kalshi.risk import Exposure, Rejected, RiskLimits, check, commit
from kalshi.trading import Order, TradingClient, TradingError, sign


def pem():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key, key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


def test_signature_verifies_with_public_key():
    key, text = pem()
    client = TradingClient(dry_run=True, env="demo", key_id="k", private_key_pem=text)
    ts = "1737654321000"
    sig = sign(client.private_key, ts, "POST", "/trade-api/v2/portfolio/orders")
    key.public_key().verify(
        base64.b64decode(sig),
        b"1737654321000POST/trade-api/v2/portfolio/orders",
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256(),
    )
    headers = client._headers("GET", "/trade-api/v2/portfolio/balance")
    assert headers["KALSHI-ACCESS-KEY"] == "k"
    assert headers["KALSHI-ACCESS-TIMESTAMP"].isdigit()


def test_order_body_and_validation():
    exp = int(time.time()) + 3600
    o = Order(
        ticker="KXNBAPTS-X-Y-20", side="no", count=7, price_cents=49, expiration_ts=exp
    )
    body = o.body()
    # Buying NO at 49c = V2 ask (sell YES) at 51c.
    assert body["side"] == "ask"
    assert body["price"] == "0.5100"
    assert body["count"] == "7.00"
    assert body["time_in_force"] == "good_till_canceled"
    assert body["self_trade_prevention_type"] == "taker_at_cross"
    assert body["post_only"] is True
    assert body["expiration_time"] == exp
    assert body["cancel_order_on_pause"] is True
    with pytest.raises(TradingError):
        Order("T", "yes", 0, 50).body()
    with pytest.raises(TradingError):
        Order("T", "yes", 1, 100).body()
    with pytest.raises(TradingError):
        Order("T", "maybe", 1, 50).body()
    with pytest.raises(TradingError):
        Order("T", "yes", 1, 50, expiration_ts=int(time.time()) - 5).body()


def test_dry_run_places_nothing():
    _, text = pem()
    client = TradingClient(dry_run=True, env="demo", key_id="k", private_key_pem=text)
    result = client.place(Order("T", "no", 3, 40))
    assert result["order_id"] == "dry-run" and result["dry_run"] is True
    assert client.log[-1]["op"] == "place" and client.log[-1]["action"] == "buy"


def test_bad_env_rejected():
    with pytest.raises(TradingError):
        TradingClient(env="staging")


def test_risk_limits():
    limits = RiskLimits(
        max_order=5, max_night=10, max_open=12, max_per_game=6, daily_loss_stop=20
    )
    ex = Exposure(by_day={}, by_game={}, open_total=0.0, realized_today=0.0)
    check(4.0, "d1", 1, limits, ex, 0)
    commit(ex, 4.0, "d1", 1)
    with pytest.raises(Rejected, match="order"):
        check(5.5, "d1", 2, limits, ex, 1)
    with pytest.raises(Rejected, match="game cap"):
        check(3.0, "d1", 1, limits, ex, 1)  # game 1 would reach 7 > 6
    check(3.0, "d1", 2, limits, ex, 1)
    commit(ex, 3.0, "d1", 2)
    with pytest.raises(Rejected, match="night cap"):
        check(4.0, "d1", 3, limits, ex, 2)  # night would reach 11 > 10
    check(5.0, "d2", 4, limits, ex, 2)  # open 7 + 5 = 12, allowed
    commit(ex, 0.5, "d2", 9)
    with pytest.raises(Rejected, match="open exposure"):
        check(5.0, "d2", 4, limits, ex, 2)  # open 7.5 + 5 = 12.5 > 12
    ex.realized_today = -20.0
    with pytest.raises(Rejected, match="loss stop"):
        check(1.0, "d2", 5, limits, ex, 2)
    ex.realized_today = 0.0
    with pytest.raises(Rejected, match="orders in one run"):
        check(1.0, "d2", 5, limits, ex, 150)


def test_pem_normalization_repairs_pasted_keys():
    from kalshi.trading import load_private_key, normalize_pem

    _, text = pem()
    one_line = text.replace("\n", " ")
    escaped = text.replace("\n", "\\n")
    assert normalize_pem(one_line) == normalize_pem(text)
    load_private_key(one_line)
    load_private_key(escaped)
    load_private_key("  " + text + "\n\n")


def test_ed25519_body_only_key():
    import base64 as b64

    from cryptography.hazmat.primitives.asymmetric import ed25519 as ed

    from kalshi.trading import load_private_key

    key = ed.Ed25519PrivateKey.generate()
    der = key.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    body = b64.b64encode(der).decode()  # what a bare paste looks like
    loaded = load_private_key(body)
    sig = sign(loaded, "1", "GET", "/trade-api/v2/portfolio/balance")
    key.public_key().verify(b64.b64decode(sig), b"1GET/trade-api/v2/portfolio/balance")


def test_per_game_order_count_cap():
    limits = RiskLimits(max_orders_per_game=2)
    ex = Exposure(by_day={}, by_game={}, open_total=0.0, realized_today=0.0)
    check(1.0, "d", 7, limits, ex, 0)
    commit(ex, 1.0, "d", 7)
    check(1.0, "d", 7, limits, ex, 1)
    commit(ex, 1.0, "d", 7)
    with pytest.raises(Rejected, match="orders already in game"):
        check(1.0, "d", 7, limits, ex, 2)
    check(1.0, "d", 8, limits, ex, 2)  # another game is fine


@pytest.mark.parametrize(
    "outcome,action,book_side,price",
    [
        ("yes", "buy", "bid", "0.4000"),
        ("yes", "sell", "ask", "0.4000"),
        ("no", "buy", "ask", "0.6000"),
        ("no", "sell", "bid", "0.6000"),
    ],
)
def test_v2_side_mapping(outcome, action, book_side, price):
    body = Order("KXNFLTEST", outcome, 2, 40, action=action).body()
    assert body["side"] == book_side
    assert body["price"] == price
    assert body["count"] == "2.00"
    assert "no_price" not in body
    assert "yes_price" not in body
    assert "action" not in body


def test_v2_demo_submission_and_cancel_uses_market_ticker(monkeypatch):
    client = TradingClient(dry_run=False, env="demo", key_id="fake")
    calls = []

    def api(method, path, params=None, body=None):
        calls.append((method, path, params, body))
        if method == "POST":
            return {
                "order_id": "o-123",
                "fill_count": "0.00",
                "remaining_count": "1.00",
            }
        if method == "DELETE":
            return {"order_id": "o-123", "reduced_by": "1.00"}
        raise AssertionError("Unexpected call")

    monkeypatch.setattr(client, "_request", api)
    order = Order("KXNFLTEST-40", "no", 1, 40)
    placed = client.place(order)
    assert placed["order_id"] == "o-123"
    assert placed["outcome_side"] == "no" and placed["action"] == "buy"
    assert calls[0][0:2] == ("POST", "/portfolio/events/orders")
    assert calls[0][3]["side"] == "ask"
    assert calls[0][3]["price"] == "0.6000"
    cancelled = client.cancel("o-123")
    assert cancelled["reduced_by"] == "1.00"
    assert calls[1][0:3] == (
        "DELETE",
        "/portfolio/events/orders/o-123",
        {"market_ticker": "KXNFLTEST-40"},
    )


def test_v2_cancel_fail_closed_without_ticker(monkeypatch):
    client = TradingClient(dry_run=False, env="demo")
    monkeypatch.setattr(client, "orders", lambda status="resting": [])
    with pytest.raises(TradingError, match="cannot route"):
        client.cancel("unknown")


def test_production_writes_require_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("ALLOW_PRODUCTION_TRADING", raising=False)
    client = TradingClient(dry_run=False, env="prod")
    def forbidden_request(*args, **kwargs):
        raise AssertionError("Must not contact production for a blocked order")

    monkeypatch.setattr(client, "_request", forbidden_request)
    with pytest.raises(TradingError, match="production order placement disabled"):
        client.place(Order("TEST", "yes", 1, 10))


def test_v2_dry_run_does_not_send_order(monkeypatch):
    client = TradingClient(dry_run=True, env="demo")
    def forbidden_request(*args, **kwargs):
        raise AssertionError("Dry run must not contact exchange")

    monkeypatch.setattr(client, "_request", forbidden_request)
    result = client.place(Order("TEST", "no", 1, 40))
    assert result["order_id"] == "dry-run"
    assert result["outcome_side"] == "no" and result["action"] == "buy"

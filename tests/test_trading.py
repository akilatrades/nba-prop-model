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
    assert body["no_price"] == 49 and body["count"] == 7 and body["action"] == "buy"
    assert (
        body["type"] == "limit"
        and body["post_only"] is True
        and body["expiration_ts"] == exp
    )
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

"""Dashboard contract tests: data classification, resilient missing values, and JS syntax."""

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

from live.summarize import live_status


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "dashboard" / "index.html"


class Ids(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])


def test_dashboard_has_unique_essential_sections_and_education():
    page = DASHBOARD.read_text()
    parser = Ids()
    parser.feed(page)
    assert len(parser.ids) == len(set(parser.ids))
    for section in (
        "overview",
        "forward",
        "paper",
        "strategies",
        "risk",
        "research-section",
        "hero-metrics",
        "forward-view",
        "system-view",
        "cumulative",
        "stats",
        "live",
    ):
        assert section in parser.ids
    for label in ("Simulated", "Volume capture", "Drawdown", "500"):
        assert label.lower() in page.lower()


def test_dashboard_inline_javascript_parses(tmp_path):
    if not shutil.which("node"):
        pytest.skip("Node.js is not installed in this environment")
    page = DASHBOARD.read_text()
    scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", page, re.S)
    assert len(scripts) == 1
    script = tmp_path / "dashboard.js"
    script.write_text(scripts[0])
    result = subprocess.run(
        ["node", "--check", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_overview_never_invents_live_profit_when_no_exchange_data(tmp_path):
    if not shutil.which("node"):
        pytest.skip("Node.js is not installed in this environment")
    harness = tmp_path / "smoke.cjs"
    harness.write_text(
        """
const fs = require("fs");
const vm = require("vm");
const html = fs.readFileSync(process.argv[2], "utf8");
const source = html.match(/<script>([\\s\\S]*?)<\\/script>/)[1];
const bootstrap = source.split('fetch("../outputs/dashboard/summary.json"')[0];
const nodes = {};
const document = {
  getElementById(id) {
    return nodes[id] || (nodes[id] = {innerHTML: "", style: {}});
  }
};
const exercise = `
overview({live: {}, live_nfl: {}, limits: {max_open: 150}});
if (!nodes["hero-metrics"].innerHTML.includes("—") ||
    !nodes["forward-view"].innerHTML.includes("Not started")) {
  throw new Error("Missing data was not represented honestly");
}
overview({
  live: {forward_validation: {
    status: "collecting", eligible_real_orders: 8,
    settled_buy_fill_events: 2, elapsed_days: 3,
    capture_volume_weighted: 0.32, qualifying_contracts: 25,
    settled_contracts: 3, cents_per_settled_contract: -2.5
  }},
  live_nfl: {},
  limits: {max_open: 150, max_order: 5, max_per_game: 25}
});
if (!nodes["hero-metrics"].innerHTML.includes("32%") ||
    !nodes["forward-view"].innerHTML.includes("2 / 500")) {
  throw new Error("Forward evidence not displayed correctly");
}
`;
vm.runInNewContext(bootstrap + exercise, {document, nodes, console});
"""
    )
    result = subprocess.run(
        ["node", str(harness), str(DASHBOARD)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_live_summary_uses_published_validation_not_fake_zeros(tmp_path):
    assert "forward_validation" not in live_status(tmp_path)
    expected = {
        "status": "collecting",
        "eligible_real_orders": 12,
        "settled_buy_fill_events": 4,
        "capture_volume_weighted": 0.34,
        "cents_per_settled_contract": -2.1,
    }
    (tmp_path / "forward_validation.json").write_text(json.dumps(expected))
    assert live_status(tmp_path)["forward_validation"] == expected


def test_invalid_validation_file_is_declared_not_silently_successful(tmp_path):
    (tmp_path / "forward_validation.json").write_text("{invalid")
    output = live_status(tmp_path)
    assert "forward_validation" not in output
    assert output.get("forward_validation_error")

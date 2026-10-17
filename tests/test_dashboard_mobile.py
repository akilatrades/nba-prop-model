"""Keep the dashboard usable on phones and preserve its evidence labels."""

import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.nav_targets = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.add(values["id"])
        if tag == "a" and values.get("href", "").startswith("#"):
            self.nav_targets.append(values["href"][1:])


def test_dashboard_mobile_scroll_does_not_reload():
    html = DASHBOARD.read_text()
    assert "location.reload(" not in html
    assert "const nextWidth = document.documentElement.clientWidth" in html
    assert "if (Math.abs(nextWidth - pageWidth) < 2) return" in html
    assert "renderCharts(latestDashboard.s, latestDashboard.pf)" in html
    assert "overflow-x:auto" in html
    assert "touch-action:pan-x pan-y" in html
    assert 'class="scroll heat-scroll" id="heat"' in html


def test_dashboard_nav_and_data_mounts_exist():
    html = DASHBOARD.read_text()
    tags = Tags()
    tags.feed(html)
    assert all(anchor in tags.ids for anchor in tags.nav_targets)
    mounted = set(re.findall(r'document.getElementById\("([^"]+)"\)', html))
    assert mounted <= tags.ids
    assert "No verified settlement results in the published logs" in html
    assert "not a real-time view of Kalshi" in html


def test_dashboard_inline_javascript_parses(tmp_path):
    node = shutil.which("node")
    if node is None:
        return  # Browsers can still parse this; CI includes Node.
    html = DASHBOARD.read_text()
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    assert len(scripts) == 1
    js = tmp_path / "dashboard.js"
    js.write_text(scripts[0])
    subprocess.run([node, "--check", str(js)], check=True)

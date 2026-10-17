"""Smoke test the published dashboard in a mobile-sized Chromium viewport.

Run explicitly after installing Playwright/Chromium; not part of regular pytest.
"""

import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    repo = Path(__file__).resolve().parents[1]
    handler = partial(SimpleHTTPRequestHandler, directory=str(repo))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                for width in (320, 375, 390, 768):
                    context = browser.new_context(
                        viewport={"width": width, "height": 780},
                        device_scale_factor=2,
                        is_mobile=True,
                        has_touch=True,
                    )
                    page = context.new_page()
                    errors = []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    url = f"http://127.0.0.1:{server.server_port}/dashboard/"
                    page.goto(url, wait_until="domcontentloaded")
                    page.wait_for_function(
                        "() => !document.querySelector('#stamp').textContent.includes('Loading')",
                        timeout=15000,
                    )
                    assert not errors, f"{width}px: JS errors: {errors}"
                    metrics = page.evaluate(
                        """() => ({
                          page: document.documentElement.clientWidth,
                          widest: document.documentElement.scrollWidth,
                          height: document.documentElement.scrollHeight,
                          navOverflow: getComputedStyle(
                            document.querySelector('.toplinks')
                          ).overflowX
                        })"""
                    )
                    assert metrics["widest"] <= metrics["page"] + 2, (
                        f"{width}px: horizontal page overflow: {metrics}"
                    )
                    if width <= 700:
                        assert metrics["navOverflow"] == "auto"
                    assert metrics["height"] > 1200, (
                        f"{width}px: unexpectedly short page; data failed to render"
                    )

                    # A phone's URL bar changing height must NOT reload the page
                    # or jump the user back to the top.
                    page.evaluate(
                        "() => { window.__scrollMarker = true; window.scrollTo(0, 650); }"
                    )
                    before = page.evaluate("window.scrollY")
                    assert before >= 200, f"{width}px: page did not scroll"
                    page.set_viewport_size({"width": width, "height": 680})
                    page.wait_for_timeout(700)
                    after = page.evaluate(
                        "() => ({y: window.scrollY, marker: window.__scrollMarker})"
                    )
                    assert after["marker"] is True, f"{width}px: page reloaded"
                    assert abs(after["y"] - before) < 35, (
                        f"{width}px: scroll jumped on height-only resize: "
                        f"{before} -> {after['y']}"
                    )
                    assert not errors, f"{width}px: JS errors after resize: {errors}"

                    # Expanding sections and jumping to lower anchors must work.
                    page.locator("details.quick-guide summary").click()
                    assert page.locator("details.quick-guide").evaluate(
                        "(el) => el.open"
                    )
                    page.locator('a[href="#research-section"]').click()
                    assert page.evaluate("location.hash") == "#research-section"
                    print(f"PASS {width}px: scroll, no reload, nav, JS and overflow")
                    context.close()
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()

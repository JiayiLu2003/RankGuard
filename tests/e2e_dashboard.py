from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    output = Path("docs/assets")
    output.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, channel="chrome")
        desktop = browser.new_page(viewport={"width": 1440, "height": 1000})
        desktop.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        desktop.goto(args.url, wait_until="networkidle")
        desktop.get_by_role(
            "heading", name="Choose what to ship. Know where it breaks."
        ).wait_for()
        assert desktop.get_by_text("REVIEW", exact=True).is_visible()
        assert desktop.locator(".model-card").count() == 4
        assert desktop.locator(".recommendation-list article").count() == 10
        desktop.screenshot(path=output / "dashboard-desktop.png", full_page=True)
        desktop_overflow = desktop.evaluate(
            "document.documentElement.scrollWidth > document.documentElement.clientWidth"
        )

        mobile = browser.new_page(viewport={"width": 375, "height": 812})
        mobile.goto(args.url, wait_until="networkidle")
        mobile.get_by_role(
            "heading", name="Choose what to ship. Know where it breaks."
        ).wait_for()
        mobile.screenshot(path=output / "dashboard-mobile.png", full_page=True)
        mobile_overflow = mobile.evaluate(
            "document.documentElement.scrollWidth > document.documentElement.clientWidth"
        )
        browser.close()

    if console_errors:
        raise AssertionError(f"Browser console errors: {console_errors}")
    if desktop_overflow or mobile_overflow:
        raise AssertionError(
            f"Horizontal overflow detected: desktop={desktop_overflow}, mobile={mobile_overflow}"
        )
    print("Dashboard E2E passed at 1440px and 375px; screenshots saved to docs/assets.")


if __name__ == "__main__":
    main()

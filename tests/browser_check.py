"""Optional live dashboard check: python tests/browser_check.py (server on port 8501)."""

import json
from datetime import UTC, datetime
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def check_dashboard():
    expect.set_options(timeout=30000)
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            channel="msedge", headless=True, args=["--enable-unsafe-swiftshader"]
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []

        def settle():
            page.wait_for_timeout(600)
            expect(page.get_by_role("img", name="Running...", exact=True)).not_to_be_visible(
                timeout=60000
            )

        page.on(
            "response",
            lambda response: (
                print("HTTP", response.status, response.url) if response.status >= 400 else None
            ),
        )
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8501")
        expect(page.get_by_text("No study results yet.", exact=False)).to_be_visible()
        page.get_by_role("button", name="Optimize cooling setpoints", exact=True).click()
        settle()
        expect(page.get_by_text("Modeled PUE", exact=True)).to_be_visible(timeout=60000)
        page.screenshot(path=str(artifacts / "dashboard-desktop.png"), full_page=True)
        page.get_by_text("3D rack profile", exact=True).click()
        expect(page.get_by_role("radio", name="3D rack profile")).to_be_checked()
        page.wait_for_function("""() => [...document.querySelectorAll('.js-plotly-plot')]
            .some(e => e.data?.some(t => t.type === 'scatter3d'))""")
        page.get_by_text("Rack thermal profile", exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(artifacts / "dashboard-3d.png"))
        page.get_by_text("2D temperatures", exact=True).click()
        page.get_by_text("Model assumptions", exact=True).click()
        expect(page.get_by_text("98% sensible IT heat;", exact=False)).to_be_visible()
        with page.expect_download() as download:
            page.get_by_role("button", name="Download study JSON", exact=True).click()
        download.value.save_as(str(artifacts / "downloaded-study.json"))
        assert json.loads((artifacts / "downloaded-study.json").read_text())["result"]["status"]
        page.get_by_role("button", name="Fetch Open-Meteo weather").click()
        expect(
            page.get_by_text("Open-Meteo observation:", exact=False).or_(
                page.get_by_text("Weather unavailable.", exact=False)
            )
        ).to_be_visible(timeout=20000)
        page.get_by_label("Dry-bulb temperature (°C)", exact=True).fill("30")
        page.get_by_label("Dry-bulb temperature (°C)", exact=True).press("Enter")
        page.get_by_label("Relative humidity (%)", exact=True).fill("60")
        page.get_by_label("Relative humidity (%)", exact=True).press("Enter")
        page.get_by_label("Electricity tariff (USD/kWh)", exact=True).fill("0.15")
        page.get_by_label("Electricity tariff (USD/kWh)", exact=True).press("Enter")
        expect(page.get_by_text("Inputs have changed.", exact=False)).to_be_visible()
        page.get_by_role("button", name="Optimize cooling setpoints", exact=True).click()
        settle()
        expect(page.get_by_text("Inputs have changed.", exact=False)).not_to_be_visible()
        expect(page.get_by_text("Modeled PUE", exact=True)).to_be_visible(timeout=60000)
        with page.expect_file_chooser() as chooser:
            page.get_by_role("button", name="Upload", exact=True).click()
        chooser.value.set_files("examples/scenario.json")
        expect(page.get_by_text("scenario.json", exact=True)).to_be_visible(timeout=15000)
        page.get_by_text("Physics models", exact=True).click()
        page.get_by_text("Resolve airflow and CRAH coil", exact=True).click()
        settle()
        page.get_by_text("Use Merkel tower model", exact=True).click()
        settle()
        page.get_by_role("button", name="Optimize cooling setpoints", exact=True).click()
        settle()
        expect(page.get_by_text("Modeled PUE", exact=True)).to_be_visible(timeout=60000)
        for label, trace_type in [("Room contours", "contour"), ("3D room", "mesh3d")]:
            page.get_by_text(label, exact=True).click()
            page.wait_for_function(
                """type => [...document.querySelectorAll('.js-plotly-plot')]
                .some(e => e.data?.some(t => t.type === type))""",
                arg=trace_type,
            )
        page.get_by_role("heading", name="Rack thermal profile").scroll_into_view_if_needed()
        page.screenshot(path=str(artifacts / "dashboard-room.png"))
        page.get_by_text("Live telemetry", exact=True).click()
        settle()
        expect(page.get_by_role("heading", name="Live advisory planning")).to_be_visible()
        with page.expect_file_chooser() as trace_chooser:
            page.get_by_role("region", name="Import telemetry trace JSON").get_by_role(
                "button", name="Upload", exact=True
            ).click()
        trace_chooser.value.set_files(
            {
                "name": "telemetry-trace.json",
                "mimeType": "application/json",
                "buffer": json.dumps(
                    {
                        "samples": [
                            {
                                "datacenter_id": "DC-WEST-01",
                                "observed_at": datetime.now(UTC).isoformat(),
                                "rack_loads_kw": {"ROW-A": 150, "ROW-B": 100},
                            }
                        ]
                    }
                ).encode(),
            }
        )
        expect(page.get_by_text("telemetry-trace.json", exact=True)).to_be_visible()
        page.get_by_role("button", name="Import telemetry trace", exact=True).click()
        settle()
        expect(page.get_by_text("Imported 1 observations.", exact=True)).to_be_visible()
        page.get_by_text("Refresh weather during live planning", exact=True).click()
        settle()
        page.get_by_text("Enable automatic advisory planning", exact=True).click()
        settle()
        expect(page.get_by_text("Economizer advisory:", exact=False)).to_be_visible(timeout=60000)
        page.screenshot(path=str(artifacts / "dashboard-live.png"))
        page.get_by_text("Enable automatic advisory planning", exact=True).click()
        settle()
        page.get_by_text("Refresh weather during live planning", exact=True).click()
        settle()
        expect(page.get_by_text("No plan for these inputs yet.", exact=False)).to_be_visible()
        expect(page.get_by_text("Economizer advisory:", exact=False)).not_to_be_visible()
        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.on("pageerror", lambda error: errors.append(str(error)))
        mobile.goto("http://127.0.0.1:8501")
        expect(mobile.get_by_role("button", name="Optimize cooling setpoints")).to_be_visible()
        mobile.get_by_role("button", name="Optimize cooling setpoints").click()
        expect(mobile.get_by_text("Modeled PUE", exact=True)).to_be_visible(timeout=60000)
        mobile.get_by_role("heading", name="DC-CoolTwin", exact=True).scroll_into_view_if_needed()
        mobile.screenshot(path=str(artifacts / "dashboard-mobile.png"), full_page=True)
        mobile.get_by_role("heading", name="Rack thermal profile").scroll_into_view_if_needed()
        mobile.screenshot(path=str(artifacts / "dashboard-mobile-thermal.png"))
        overflow = mobile.evaluate("document.documentElement.scrollWidth > window.innerWidth")
        assert not overflow, "Page overflows mobile viewport"
        mobile.keyboard.press("Tab")
        assert mobile.evaluate("document.activeElement.tagName") != "BODY"
        assert not errors, errors
        browser.close()
        print(
            "PASS: optimization, thermal views, assumptions, download, weather, ambient inputs, "
            "tariff, JSON upload, detailed physics, room views, trace import, live planning, "
            "mobile overflow, keyboard focus; no page errors"
        )


if __name__ == "__main__":
    check_dashboard()

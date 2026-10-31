#!/usr/bin/env python3
"""Exercise an existing offline gallery in CPU Chromium; never runs inference.

Requires the optional ``playwright`` package and an installed Chromium browser.
The report distinguishes passed checks, failures and unsupported controls. Network
requests are blocked and recorded. Optional screenshots are review artifacts,
not experimental predictions. Existing reports/screenshots are not overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse, unquote

CPU_BROWSER_ARGS = ["--disable-gpu", "--use-gl=angle", "--use-angle=swiftshader"]


def launch_chromium(playwright):
    return playwright.chromium.launch(headless=True, args=CPU_BROWSER_ARGS)


def _frame(page):
    page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")


def _check(report, name, fn):
    try:
        details = fn()
        report["checks"].append({"name": name, "status": "passed", "details": details})
    except Exception as exc:
        report["checks"].append({"name": name, "status": "failed", "error": str(exc)})


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _not_applicable(report, name, reason):
    report["checks"].append({"name": name, "status": "not_applicable", "reason": reason})


def _selected(page):
    return page.evaluate("""() => {
      const d=JSON.parse(document.getElementById('scene-data').textContent);
      return [...document.querySelectorAll('.view-head select')].map(s=>d.panels[Number(s.value)]);
    }""")


def _scene_checks(page, report, *, require_steps=False, require_models=0, screenshots=None):
    page.wait_for_selector("#scene-title")
    page.wait_for_function("document.querySelectorAll('#workspace canvas').length===2")
    _frame(page)
    data = json.loads(page.locator("#scene-data").text_content())
    report["scene_id"] = data["scene_id"]
    report["input_sha256"] = hashlib.sha256(page.locator("#scene-data").text_content().encode()).hexdigest()
    options = page.locator("#model-filter option").evaluate_all("els=>els.map(e=>e.value).filter(Boolean)")
    selects = page.locator(".view-head select")

    def model_selection():
        _assert(len(options) >= require_models, f"Need {require_models} model choices; found {len(options)}")
        for name in options:
            page.select_option("#model-filter", name)
            _frame(page)
            _assert(_selected(page)[0]["metadata"].get("model_name") == name,
                    f"Selecting {name!r} did not update panel A")
            visible = page.locator(".panel-chip:not([hidden])").count()
            expected = sum(p["metadata"].get("model_name") == name for p in data["panels"])
            _assert(visible == expected, f"Model library filter disagrees for {name!r}")
            _assert(page.locator('.panel-chip[hidden]').evaluate_all("els=>els.every(e=>getComputedStyle(e).display==='none')"),
                    f"Hidden model panels remain visible for {name!r}")
        page.select_option("#model-filter", "")
        return {"models_exercised": options}
    if options or require_models:
        _check(report, "model_checkpoint_selection", model_selection)
    else:
        _not_applicable(report, "model_checkpoint_selection", "No model-labelled panels")

    # Check initial camera settings through public controls, before modifying them.
    def declared_camera():
        expected_size = data["camera"].get("point_size", 2)
        actual_size = float(page.locator("#size").input_value())
        _assert(abs(actual_size - expected_size) < 1e-6,
                f"Viewer point size {actual_size} differs from recorded {expected_size}")
        clip = data["camera"].get("clip_z")
        if clip is not None and data["z_range"][1] > data["z_range"][0]:
            lo, hi = data["z_range"]
            expected_height = min(100., max(0., 100 * (clip - lo) / (hi - lo)))
            actual_height = float(page.locator("#height").input_value())
            _assert(abs(actual_height - expected_height) < 0.51,
                    f"Viewer height {actual_height}% differs from recorded clip {expected_height}%")
        return {"point_size": actual_size, "recorded_clip_z": clip}
    _check(report, "declared_camera_initialization", declared_camera)

    step_models = {}
    for i, panel in enumerate(data["panels"]):
        meta = panel["metadata"]
        if isinstance(meta.get("step"), int) and not meta.get("failure_type"):
            step_models.setdefault(meta.get("model_name"), {}).setdefault(meta["step"], i)
    paired = [name for name in options if name in step_models]
    if paired and len(paired) >= 2:
        common = sorted(set(step_models[paired[0]]) & set(step_models[paired[1]]))
    else:
        common = []
    def step_selection():
        _assert(len(common) >= 2, "Need at least two shared saved steps for two models")
        names = paired[:2]
        for view, name in enumerate(names):
            selects.nth(view).select_option(str(step_models[name][common[0]]))
        page.select_option("#model-filter", names[0])
        for step in common:
            page.locator("#interaction-step").evaluate("(e,v)=>{e.value=v;e.dispatchEvent(new Event('input',{bubbles:true}))}", str(step))
            _frame(page)
            for view, name in enumerate(names):
                selected = _selected(page)[view]
                _assert(selected["metadata"].get("model_name") == name,
                        f"Step selection replaced panel {view}'s model")
                _assert(selected["metadata"].get("step") == step,
                        f"Panel {view} did not move to saved step {step}")
                _assert(page.locator(".caption").nth(view).text_content() == selected["caption"],
                        f"Panel {view} caption was not updated with its selected step")
        return {"models": names, "shared_steps_exercised": common,
                "claim": "Controls select saved snapshots; identical clicks are not assumed"}
    if len(common) >= 2 or require_steps:
        _check(report, "paired_interaction_steps", step_selection)
    else:
        _not_applicable(report, "paired_interaction_steps", "Two models with two shared saved steps are unavailable")

    objects = page.locator("#object-filter option").evaluate_all("els=>els.map(e=>e.value).filter(Boolean)")
    if objects:
        def object_inspection():
            value = objects[-1]
            page.select_option("#object-filter", value)
            _frame(page)
            text = page.locator("#point-info").text_content()
            _assert("Original point ID:" in text and f"Instance label: {value}" in text,
                    "Object inspection does not identify an original point and selected GT label")
            point_id = int(text.split("Original point ID: ", 1)[1].splitlines()[0])
            packed_index = data["indices"].index(point_id)
            _assert(data["labels"][packed_index] == int(value), "Inspector original-point correspondence differs")
            page.select_option("#object-filter", "")
            return {"selected_label": int(value), "original_point_id": point_id}
        _check(report, "object_original_point_inspection", object_inspection)
    else:
        _not_applicable(report, "object_original_point_inspection", "No sampled ground-truth object labels")

    def linked_camera():
        # Identical layers and equal-size canvases must render identically, both
        # before and after a keyboard orbit sent to only the left canvas.
        page.select_option("#model-filter", "")
        for i in range(2):
            selects.nth(i).select_option("0")
        page.check("#compare")
        page.click("#reset")
        _frame(page)
        canvases = page.locator("#workspace canvas")
        before = [canvases.nth(i).evaluate("c=>c.toDataURL()") for i in range(2)]
        _assert(before[0] == before[1], "Linked panels disagree before orbit with identical layers")
        canvases.nth(0).focus()
        page.keyboard.press("ArrowRight")
        page.keyboard.press("ArrowUp")
        _frame(page)
        after = [canvases.nth(i).evaluate("c=>c.toDataURL()") for i in range(2)]
        _assert(after[0] == after[1], "Linked panels disagree after orbiting only panel A")
        _assert(before[0] != after[0], "Camera orbit did not change either rendered canvas")
        with page.expect_download() as event:
            page.click("#save-camera")
        camera = json.loads(Path(event.value.path()).read_text())
        _assert(camera.get("schema") == "delimit3d_view_camera/v1", "Camera download has wrong schema")
        _assert(camera["scene_id"] == data["scene_id"] and camera["center"] == data["center"],
                "Camera download lost scene/frame identity")
        _assert(camera["yaw"] != data["camera"]["yaw"], "Camera download omitted the performed orbit")
        _assert(camera["projection"] == "orthographic" and camera["up_axis"] == "z", "Camera convention changed")
        return {"download": camera, "linked_canvas_sha256": hashlib.sha256(after[0].encode()).hexdigest()}
    _check(report, "linked_camera_orbit_and_download", linked_camera)
    def capture_export():
        downloads=[]
        def on_download(download):downloads.append(download)
        page.on('download',on_download)
        try:
            page.click('#export-view')
            for _ in range(100):
                if len(downloads)>=2:break
                page.wait_for_timeout(100)
            _assert(len(downloads)==2,'Current view must download PNG and matching JSON')
            sidecar=next(d for d in downloads if d.suggested_filename.endswith('.json'))
            capture=json.loads(Path(sidecar.path()).read_text())
            png=next(d for d in downloads if d.suggested_filename.endswith('.png'))
            _assert(capture['image']['sha256']==hashlib.sha256(Path(png.path()).read_bytes()).hexdigest(),'Capture PNG does not match sidecar hash')
            _assert(capture['schema']=='delimit3d_point_view_capture/v1','Wrong capture schema')
            _assert(capture['payload_sha256']==hashlib.sha256(page.locator('#scene-data').text_content().encode()).hexdigest(),'Capture source payload hash differs')
            _assert([p['key'] for p in capture['panels']]==[p['key'] for p in _selected(page)],'Capture panel identities differ')
            _assert(capture['original_point_ids']==data['indices'],'Capture lost original-point sample identities')
            _assert(capture['camera']['yaw']!=data['camera'].get('yaw',-.68),'Capture lost the interactive orbit')
            return {'files':[d.suggested_filename for d in downloads],'payload_sha256':capture['payload_sha256'],'claim':'Exploratory capture with numeric scales and exact current state'}
        finally:page.remove_listener('download',on_download)
    _check(report,'current_view_png_and_research_record',capture_export)
    if screenshots:
        screenshot = screenshots / (hashlib.sha256(page.url.encode()).hexdigest()[:12] + ".png")
        page.screenshot(path=str(screenshot), full_page=False, timeout=60000)
        report["review_screenshot"] = str(screenshot)
        report["review_screenshot_scope"] = "current viewport; controls below the fold are exercised programmatically"


def _index_checks(page, report):
    cards = page.locator("article[data-dataset]").evaluate_all("els=>els.map(e=>({...e.dataset}))")
    def filtering():
        for selector, key in [("dataset-filter", "dataset"), ("experiment-filter", "experiment"), ("tag-filter", "tags"), ("failure-filter", "failures")]:
            choices = page.locator(f"#{selector} option").evaluate_all("els=>els.map(e=>e.value).filter(Boolean)")
            for choice in choices:
                page.select_option(f"#{selector}", choice)
                actual = page.locator("article[data-dataset]:not([hidden])").count()
                expected = sum(choice in card[key].split() if key in {"tags", "failures"} else choice == card[key] for card in cards)
                _assert(actual == expected, f"{selector}={choice!r}: visible {actual}, expected {expected}")
            page.select_option(f"#{selector}", "")
        _assert(page.locator("article[data-dataset]:not([hidden])").count() == len(cards), "Reset filters did not restore collection")
        return {"scene_cards": len(cards), "controls": ["dataset", "experiment", "case_type", "failure"]}
    _check(report, "collection_filters", filtering)


def check_gallery(path, *, browser=None, require_steps=False, require_models=0, screenshots=None):
    """Return a JSON-safe report for an existing scene or collection index.

    Supplying a browser permits fixture tests to reuse one CPU-only process.
    ``not_applicable`` never means that an unavailable feature was verified.
    """
    path = Path(path).resolve()
    if path.is_dir():
        path /= "index.html"
    if not path.is_file():
        raise FileNotFoundError(path)
    if screenshots:
        screenshots = Path(screenshots).resolve()
        screenshots.mkdir(parents=True, exist_ok=False)
    if browser is None:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as runtime:
            owned = launch_chromium(runtime)
            try:
                return _check_with_browser(path, owned, require_steps, require_models, screenshots)
            finally:
                owned.close()
    return _check_with_browser(path, browser, require_steps, require_models, screenshots)


def _check_with_browser(path, browser, require_steps, require_models, screenshots):
    result = {"schema": "delimit3d_browser_review/v1", "checked_at": datetime.now(timezone.utc).isoformat(),
              "input": str(path), "browser": browser.version, "cpu_browser_flags": CPU_BROWSER_ARGS,
              "network_policy": "offline context; external requests intercepted and aborted", "pages": []}
    context = browser.new_context(offline=True, viewport={"width": 1440, "height": 1000}, accept_downloads=True)
    external = []
    def block_external(route):
        if urlparse(route.request.url).scheme in {"http", "https", "ws", "wss"}:
            external.append(route.request.url)
            route.abort()
        else:
            route.continue_()
    context.route("**/*", block_external)
    queue, visited = [path.as_uri()], set()
    try:
        while queue:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            page = context.new_page()
            record = {"url": url, "checks": [], "browser_errors": [], "failed_requests": []}
            page.on("pageerror", lambda error, r=record: r["browser_errors"].append(str(error)))
            page.on("console", lambda message, r=record: r["browser_errors"].append(message.text) if message.type == "error" else None)
            page.on("requestfailed", lambda request, r=record: r["failed_requests"].append({"url": request.url, "failure": request.failure}))
            try:
                page.goto(url, wait_until="load", timeout=60000)
                if page.locator("#scene-data").count():
                    _scene_checks(page, record, require_steps=require_steps, require_models=require_models, screenshots=screenshots)
                elif page.locator("#dataset-filter").count():
                    _index_checks(page, record)
                    for link in page.locator("a").evaluate_all("els=>els.filter(e=>e.textContent==='Explore 3D').map(e=>e.href)"):
                        parsed = urlparse(link)
                        _assert(parsed.scheme == "file", "Collection scene link is not offline")
                        queue.append(Path(unquote(parsed.path)).resolve().as_uri())
                else:
                    raise ValueError("Input is not a supported suite point gallery or collection")
            except Exception as exc:
                record["checks"].append({"name": "page_loading", "status": "failed", "error": str(exc)})
            _check(record, "no_browser_errors", lambda: _assert(not record["browser_errors"], str(record["browser_errors"])))
            _check(record, "local_assets_loaded", lambda: _assert(not record["failed_requests"], str(record["failed_requests"])))
            result["pages"].append(record)
            page.close()
    finally:
        context.close()
    result["external_requests"] = external
    result["status"] = "failed" if external or any(c["status"] == "failed" for p in result["pages"] for c in p["checks"]) else "passed"
    result["limitations"] = ["Automated point-gallery controls only; mesh WebGL controls and scientific figure appearance require separate review"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gallery", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screenshots", type=Path)
    parser.add_argument("--require-steps", action="store_true")
    parser.add_argument("--require-models", type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    report = check_gallery(args.gallery, require_steps=args.require_steps,
                           require_models=args.require_models, screenshots=args.screenshots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "report": str(args.output), "pages": len(report["pages"])}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline interactive galleries and matched publication views of point clouds.

The HTML viewer intentionally uses the browser's Canvas API only. Every payload,
style and script is embedded so galleries remain usable on a disconnected machine
and through ``file://``. Matplotlib is imported only for static exports.
"""

from __future__ import annotations

import html
import json
import math
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .contracts import Panel, Scene


_YAW = -0.68
_PITCH = 0.62


def _safe_stem(key: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", key).strip("._")
    return stem or "panel"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _script_json(payload: dict[str, Any]) -> str:
    # An HTML parser recognizes </script> even inside a JavaScript string.
    return (
        json.dumps(payload, separators=(",", ":"), default=_json_default, allow_nan=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _rgb(colors: np.ndarray) -> np.ndarray:
    array = np.asarray(colors)
    if not np.isfinite(array).all():
        raise ValueError("Panel colors must be finite RGB values")
    if np.issubdtype(array.dtype, np.floating) and array.size and array.max() <= 1.0:
        array = array * 255.0
    return np.rint(np.clip(array, 0, 255)).astype(np.uint8)


def _compact_metadata(value: Any, depth: int = 0) -> Any:
    """Keep records useful without embedding unsampled token/feature arrays."""
    if depth > 6:
        return "[nested metadata omitted]"
    if isinstance(value, np.ndarray):
        if value.size > 32:
            return {"shape": list(value.shape), "dtype": str(value.dtype),
                    "note": "Array omitted from gallery; see the experiment record"}
        return _compact_metadata(value.tolist(), depth + 1)
    if isinstance(value, dict):
        return {str(key): _compact_metadata(item, depth + 1)
                for key, item in list(value.items())[:80]}
    if isinstance(value, (list, tuple)):
        if len(value) > 64:
            return {"length": len(value), "note": "Sequence omitted from gallery"}
        return [_compact_metadata(item, depth + 1) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, str) and len(value) > 4000:
        return value[:4000] + " [truncated]"
    return value


def _project(points: np.ndarray, center: np.ndarray, camera: dict | None = None) -> np.ndarray:
    if camera is not None:
        from .recipes import project
        return project(points, {"center": center, **camera})
    xyz = np.asarray(points, dtype=np.float64) - center
    cy, sy = math.cos(_YAW), math.sin(_YAW)
    cp, sp = math.cos(_PITCH), math.sin(_PITCH)
    horizontal = cy * xyz[:, 0] - sy * xyz[:, 1]
    away = sy * xyz[:, 0] + cy * xyz[:, 1]
    return np.column_stack((horizontal, cp * xyz[:, 2] - sp * away,
                            cp * away + sp * xyz[:, 2]))


def _legend(panel: Panel, colors: np.ndarray) -> dict[str, Any] | None:
    if panel.values is None or panel.limits is None:
        return None
    color_stops = panel.metadata.get("color_stops")
    if color_stops is not None:
        stops = np.asarray(color_stops)
        if stops.ndim != 2 or stops.shape[1] != 3 or len(stops) < 2:
            raise ValueError(f"Panel {panel.key!r}: color_stops must be an M x 3 RGB palette")
        return {"limits": list(panel.limits), "colors": _rgb(stops).tolist(),
                "label": panel.metadata.get("value_label", "Value"),
                "palette_source": "fixed color stops"}
    values = np.asarray(panel.values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values)
    if not valid.any():
        return None
    # Read the actual panel palette instead of assuming a matplotlib palette.
    order = np.argsort(values[valid], kind="stable")
    sorted_values, sorted_colors = values[valid][order], colors[valid][order]
    ticks = np.linspace(float(panel.limits[0]), float(panel.limits[1]), 24)
    positions = np.searchsorted(sorted_values, ticks).clip(0, len(sorted_values) - 1)
    return {"limits": list(panel.limits), "colors": sorted_colors[positions].tolist(),
            "label": panel.metadata.get("value_label", "Value") + " (observed colors)",
            "palette_source": "approximate palette inferred from observed values"}


def _click_markers(scene: Scene, panel: Panel) -> list[dict[str, Any]]:
    markers = []
    for ordinal, click in enumerate(panel.metadata.get("clicks", []), 1):
        index = click.get("point_index", click.get("index"))
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(scene.points):
            raise ValueError(f"Panel {panel.key!r} contains an invalid click point index")
        positive = click.get("is_positive", click.get("positive"))
        polarity = "unspecified"
        if isinstance(positive, (bool, np.bool_)):
            polarity = "positive" if positive else "negative"
        elif str(click.get("polarity", click.get("kind", ""))).lower() in {"positive", "negative", "background"}:
            polarity = str(click.get("polarity", click.get("kind"))).lower()
        elif "object_id" in click:
            polarity = "background" if click["object_id"] == 0 else "object"
        markers.append({"index": int(index), "point": np.asarray(scene.points[index]).tolist(),
                        "ordinal": ordinal, "polarity": polarity,
                        "object_id": click.get("object_id")})
    return markers


def _payload(scene: Scene, panels: list[Panel], indices: np.ndarray, seed: int) -> dict[str, Any]:
    points = np.asarray(scene.points, dtype=np.float64)
    center = np.asarray(scene.metadata.get("camera", {}).get("center",
                        (points.min(axis=0) + points.max(axis=0)) / 2.0))
    radius = max(float(np.linalg.norm(points - center, axis=1).max()), 1e-6)
    packed_panels = []
    for panel in panels:
        colors = _rgb(panel.colors)
        values = None
        if panel.values is not None:
            array = np.asarray(panel.values, dtype=np.float64).reshape(-1)[indices]
            undefined = panel.metadata.get("undefined_mask")
            if undefined is None:
                invalid = np.zeros(len(indices), dtype=bool)
            else:
                mask = np.asarray(undefined, dtype=bool)
                if mask.shape != (len(points),):
                    raise ValueError(f"Panel {panel.key!r}: undefined_mask must align with scene points")
                invalid = mask[indices]
            values = [float(value) if np.isfinite(value) and not invalid[i] else None
                      for i, value in enumerate(array)]
        query = None
        if panel.query_index is not None:
            qi = int(panel.query_index)
            query = {"index": qi, "point": points[qi].tolist()}
        packed_panels.append({"key": panel.key, "title": panel.title,
                              "caption": panel.caption, "colors": colors[indices].tolist(),
                              "values": values, "legend": _legend(panel, colors),
                              "query": query, "clicks": _click_markers(scene, panel),
                              "metadata": _compact_metadata(panel.metadata)})
    return {"scene_id": scene.scene_id, "points": points[indices].tolist(),
            "indices": indices.tolist(), "center": center.tolist(), "radius": radius,
            "z_range": [float(points[:, 2].min()), float(points[:, 2].max())],
            "total_points": len(points), "seed": seed,
            "labels": None if scene.labels is None else np.asarray(scene.labels)[indices].tolist(),
            "semantic_labels": (None if scene.semantic_labels is None else
                                np.asarray(scene.semantic_labels)[indices].tolist()),
            "metadata": _compact_metadata(scene.metadata),
            "panels": packed_panels, "camera": scene.metadata.get("camera", {"yaw": _YAW, "pitch": _PITCH})}


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        from matplotlib.cm import ScalarMappable
    except ImportError as exc:
        raise RuntimeError(
            "PNG/PDF export needs matplotlib. Install with `pip install '.[plots]'` "
            "from the Delimit3D checkout, or request formats=('html',) for an offline gallery."
        ) from exc
    return plt, LinearSegmentedColormap, Normalize, ScalarMappable


def _static_exports(scene: Scene, panels: list[Panel], output_dir: Path,
                    indices: np.ndarray, formats: tuple[str, ...], stems: list[str],
                    payload: dict[str, Any]) -> dict[str, Any]:
    plt, LinearSegmentedColormap, Normalize, ScalarMappable = _matplotlib()
    center = np.asarray(payload["center"])
    clip_z = payload["camera"].get("clip_z")
    if clip_z is not None:
        indices = indices[np.asarray(scene.points)[indices,2] <= clip_z]
    full_projection = _project(np.asarray(scene.points), center, payload["camera"])
    lo, hi = full_projection[:, :2].min(axis=0), full_projection[:, :2].max(axis=0)
    span = hi - lo
    margin = np.maximum(span, max(float(span.max()), 1e-6) * 0.08) * 0.06
    limits = np.column_stack((lo - margin, hi + margin))
    projected = _project(np.asarray(scene.points)[indices], center, payload["camera"])
    order = np.argsort(projected[:, 2], kind="stable")
    xy = projected[order, :2]
    result: dict[str, Any] = {fmt: [] for fmt in formats}

    def draw(ax, panel, *, title_size=12, show_caption=True):
        colors = _rgb(panel.colors)[indices][order] / 255.0
        ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=payload["camera"].get("point_size",3.5), linewidths=0,
                   rasterized=True, marker="o")
        if panel.query_index is not None and (clip_z is None or scene.points[panel.query_index,2] <= clip_z):
            point = _project(np.asarray(scene.points)[[int(panel.query_index)]], center, payload["camera"])[0]
            ax.scatter([point[0]], [point[1]], s=110, marker="*", c="#ffcc50",
                       edgecolors="#142233", linewidths=0.9, zorder=10)
        for click in _click_markers(scene, panel):
            if clip_z is not None and click["point"][2] > clip_z:continue
            point = _project(np.asarray([click["point"]]), center, payload["camera"])[0]
            color = ("#bd4646" if click["polarity"] in {"negative", "background"} else
                     "#208866" if click["polarity"] == "positive" else "#326da8")
            ax.scatter([point[0]], [point[1]], s=95, marker="x" if click["polarity"] in {"negative", "background"} else "o", c=color,
                       edgecolors="white", linewidths=1, zorder=11)
            ax.text(point[0], point[1], str(click["ordinal"]), ha="center", va="center",
                    color="white", fontsize=6, zorder=12)
        ax.set(xlim=tuple(limits[0]), ylim=tuple(limits[1]), aspect="equal")
        ax.axis("off")
        ax.set_title(textwrap.fill(panel.title, 42), fontsize=title_size,
                     fontweight="semibold", color="#192b41", pad=8)
        if show_caption and panel.caption:
            ax.text(0.5, -0.22, textwrap.fill(panel.caption, 80), ha="center", va="top",
                    transform=ax.transAxes, fontsize=7, color="#526075")
        legend = _legend(panel, _rgb(panel.colors))
        if legend is not None:
            cmap = LinearSegmentedColormap.from_list(
                "panel_palette", np.asarray(legend["colors"]) / 255.0)
            lo, hi = legend["limits"]
            if hi > lo:
                mappable = ScalarMappable(norm=Normalize(vmin=lo, vmax=hi), cmap=cmap)
                cax = ax.inset_axes([0.18, -0.105 if show_caption else -0.02, 0.64, 0.021])
                colorbar = ax.figure.colorbar(mappable, cax=cax, orientation="horizontal")
                colorbar.set_ticks([lo, (lo + hi) / 2, hi])
                colorbar.ax.tick_params(labelsize=6, length=2, pad=1)
                colorbar.outline.set_visible(False)
                colorbar.set_label(str(legend["label"]), fontsize=6, labelpad=1)

    illustrative = bool(scene.metadata.get("illustrative_only", False))
    kind = "synthetic example" if scene.metadata.get("synthetic") else "scene"
    note = (f"Illustrative {kind}; not benchmark evidence. " if illustrative else "")
    note += (f"{len(indices):,} / {len(scene.points):,} points · shared camera and scale · "
             "star = query point; numbered circles = recorded clicks")
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.facecolor": "white"}):
        for panel, stem in zip(panels, stems):
            fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="white")
            fig.subplots_adjust(left=0.07, right=0.93, top=0.86, bottom=0.30)
            draw(ax, panel)
            fig.text(0.075, 0.95, "DELIMIT3D / REPRESENTATION ATLAS", fontsize=7,
                     color="#537080", fontweight="bold")
            fig.text(0.075, 0.055, textwrap.fill(f"{scene.scene_id} · {note}", 105),
                     fontsize=6.5, color="#526075")
            for fmt in formats:
                filename = f"{stem}.{fmt}"
                fig.savefig(output_dir / filename, dpi=240, bbox_inches="tight")
                result[fmt].append(filename)
            plt.close(fig)

        # Paginate large suites instead of producing an unreadably tall contact sheet.
        pages = max(1, math.ceil(len(panels) / 6))
        contact_files = {fmt: [] for fmt in formats}
        for page in range(pages):
            subset = panels[page * 6:(page + 1) * 6]
            columns = min(3, len(subset))
            rows = math.ceil(len(subset) / columns)
            fig, axes = plt.subplots(rows, columns, figsize=(16, 9), squeeze=False,
                                     facecolor="white")
            fig.subplots_adjust(left=0.035, right=0.97, top=0.85, bottom=0.11,
                                wspace=0.10, hspace=0.38)
            for ax, panel in zip(axes.flat, subset):
                draw(ax, panel, title_size=11, show_caption=False)
            for ax in list(axes.flat)[len(subset):]:
                ax.set_visible(False)
            fig.text(0.045, 0.945, "DELIMIT3D / REPRESENTATION ATLAS", fontsize=10,
                     color="#537080", fontweight="bold")
            fig.text(0.045, 0.89, str(scene.scene_id), fontsize=20, color="#192b41")
            fig.text(0.045, 0.035, textwrap.fill(note, 150), fontsize=8, color="#526075")
            fig.text(0.96, 0.935, f"{page + 1:02d} / {pages:02d}", ha="right",
                     fontsize=10, color="#537080")
            for fmt in formats:
                suffix = "" if page == 0 else f"_{page + 1:02d}"
                filename = f"contact_sheet{suffix}.{fmt}"
                fig.savefig(output_dir / filename, dpi=200)
                contact_files[fmt].append(filename)
            plt.close(fig)
        for fmt, filenames in contact_files.items():
            result[f"contact_sheet_{fmt}"] = filenames[0]
            result[f"contact_sheets_{fmt}"] = filenames
    return result


def render_scene(scene: Scene, panels: list[Panel], output_dir: Path, *,
                 sample_indices: np.ndarray,
                 formats: tuple[str, ...] = ("html", "png", "pdf"),
                 seed: int = 0) -> dict[str, Any]:
    """Write one scene's self-contained gallery and publication figure bundle.

    All panels use exactly ``sample_indices`` and a scene-wide center/radius.
    Query markers retain their original point IDs, even if called with a sample
    that omits the query. Returned filenames are relative to ``output_dir``.
    """
    formats = tuple(dict.fromkeys(formats))
    unknown = set(formats) - {"html", "png", "pdf"}
    if unknown:
        raise ValueError(f"Unsupported render formats: {', '.join(sorted(unknown))}")
    points = np.asarray(scene.points)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("Scene points must be a nonempty N x 3 array")
    if not np.isfinite(points).all():
        raise ValueError("Scene coordinates must be finite before visualization")
    if not panels:
        raise ValueError("At least one visualization panel is required")
    raw_indices = np.asarray(sample_indices)
    if raw_indices.ndim != 1 or raw_indices.dtype.kind not in "iu":
        raise ValueError("sample_indices must be a one-dimensional integer array")
    indices = raw_indices.astype(np.int64, copy=False)
    if not len(indices) or indices.min() < 0 or indices.max() >= len(points):
        raise ValueError("sample_indices must contain valid original point indices")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("sample_indices must not contain duplicate point indices")
    stems = [_safe_stem(panel.key) for panel in panels]
    if len(set(stems)) != len(stems) or any(stem.startswith("contact_sheet") for stem in stems):
        raise ValueError("Panel keys must have unique, non-reserved filename forms")
    for panel in panels:
        if np.asarray(panel.colors).shape != (len(points), 3):
            raise ValueError(f"Panel {panel.key!r} must have one RGB color per scene point")
        if panel.values is not None and np.asarray(panel.values).size != len(points):
            raise ValueError(f"Panel {panel.key!r} values do not align with scene points")
        if panel.query_index is not None and (
            not isinstance(panel.query_index, (int, np.integer))
            or not 0 <= int(panel.query_index) < len(points)
        ):
            raise ValueError(f"Panel {panel.key!r} query index is outside the scene")
        if panel.limits is not None and (
            len(panel.limits) != 2 or not np.isfinite(panel.limits).all()
            or panel.limits[0] > panel.limits[1]
        ):
            raise ValueError(f"Panel {panel.key!r} requires finite ordered scalar limits")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _payload(scene, panels, indices, seed)
    result: dict[str, Any] = {}
    static = tuple(fmt for fmt in formats if fmt in {"png", "pdf"})
    if static:
        result.update(_static_exports(scene, panels, output_dir, indices, static, stems, payload))
    if "html" in formats:
        payload["files"] = result
        payload["stems"] = stems
        document = _HTML.replace("__TITLE__", html.escape(str(scene.scene_id)))
        document = document.replace("__PAYLOAD__", _script_json(payload))
        (output_dir / "index.html").write_text(document, encoding="utf-8")
        result["html"] = "index.html"
    return result


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · Delimit3D representation atlas</title>
<style>
:root{color-scheme:light;--bg:#edf1f4;--surface:#fff;--ink:#172b40;--muted:#576d80;--line:#dce4ea;--accent:#147e89;--canvas:#f5f7fa;--shadow:0 9px 36px #23394e08}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}body.dark{color-scheme:dark;--bg:#101b28;--surface:#192738;--ink:#e3eef7;--muted:#a2b5c6;--line:#304253;--accent:#67d3d1;--canvas:#122031;--shadow:none}button,select,input{font:inherit}button,select{background:var(--surface);color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:7px 10px}button{cursor:pointer}button:hover{border-color:var(--accent)}button:focus-visible,select:focus-visible,input:focus-visible,canvas:focus-visible{outline:2px solid var(--accent);outline-offset:2px}a{color:var(--accent)}header{max-width:1560px;padding:31px 32px 20px;margin:auto}.eyebrow{text-transform:uppercase;letter-spacing:.19em;font-size:10px;font-weight:750;color:var(--accent)}h1{font-size:clamp(25px,3vw,37px);letter-spacing:-.035em;line-height:1.15;margin:12px 0 9px;overflow-wrap:anywhere}header p{max-width:900px;color:var(--muted);margin:0}.topline{display:flex;justify-content:space-between;align-items:center;gap:12px}.badges{display:flex;gap:8px;flex-wrap:wrap;margin-top:17px}.badge{font-size:11px;color:var(--muted);background:var(--surface);border:1px solid var(--line);border-radius:30px;padding:5px 11px}.warning{color:#835614;background:#fff2d3;border-color:#efdeb6}.toolbar{display:flex;gap:16px;align-items:center;flex-wrap:wrap;padding:15px 18px;background:var(--surface);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}.control{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted)}.control input[type=range]{width:100px;accent-color:var(--accent)}.control input[type=checkbox]{accent-color:var(--accent)}main{max-width:1560px;margin:auto;padding:0 32px 36px}.workspace{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}.workspace.single{grid-template-columns:1fr}.view{background:var(--surface);border:1px solid var(--line);border-radius:13px;overflow:hidden;box-shadow:var(--shadow)}.view-head{padding:15px 17px 13px;display:flex;align-items:center;justify-content:space-between;gap:12px}.view-head select{max-width:calc(100% - 44px);font-weight:600;font-size:13px}.slot{color:var(--accent);font-size:11px;font-weight:750;letter-spacing:.1em}.stage{position:relative;background:var(--canvas)}canvas{display:block;width:100%;height:440px;cursor:grab;touch-action:none}canvas:active{cursor:grabbing}.hint{position:absolute;left:15px;bottom:12px;color:var(--muted);font-size:10px;pointer-events:none}.panel-info{min-height:105px;padding:12px 17px 16px}.caption{font-size:12px;color:var(--muted);margin:0 0 10px;min-height:18px;overflow-wrap:anywhere}.legend{width:220px;max-width:100%;font-size:10px;color:var(--muted)}.gradient{height:8px;border-radius:3px;margin:4px 0}.ticks{display:flex;justify-content:space-between}.downloads{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0 2px;font-size:11px}.inspect{margin-top:18px;display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{border:1px solid var(--line);background:var(--surface);border-radius:12px;padding:18px;min-width:0}.card h2{font-size:12px;letter-spacing:.025em;margin:0 0 10px;font-weight:700}.card p{font-size:12px;color:var(--muted);margin:0;overflow-wrap:anywhere}#point-info{font-variant-numeric:tabular-nums;white-space:pre-wrap}.panel-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(225px,1fr));gap:10px;margin-top:12px}.panel-chip{text-align:left;padding:12px;font-size:12px;display:flex;flex-direction:column;gap:5px}.panel-chip small{color:var(--muted);font-size:10px}.panel-chip.active{border-color:var(--accent);box-shadow:inset 3px 0 var(--accent)}.library{margin-top:26px}.library h2{font-size:15px}.library p{color:var(--muted);font-size:12px}details{margin-top:18px;font-size:12px;color:var(--muted)}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;border:1px solid var(--line);padding:14px;border-radius:8px;max-height:300px;overflow:auto;font-size:10px}footer{font-size:11px;color:var(--muted);margin-top:28px}#tooltip{position:fixed;display:none;pointer-events:none;background:var(--ink);color:var(--surface);font-size:11px;padding:6px 9px;border-radius:5px;z-index:10;white-space:pre-line}.hidden{display:none!important}@media(max-width:900px){header{padding:23px 18px 18px}main{padding:0 18px 25px}.workspace,.inspect{grid-template-columns:1fr}canvas{height:380px}.toolbar{gap:11px}.view-head select{font-size:12px}}@media print{.toolbar,.hint,.library,details,#theme{display:none}body{background:#fff}.workspace{grid-template-columns:1fr 1fr}canvas{height:330px}header{padding-top:10px}.view,.card{break-inside:avoid}}
[hidden]{display:none!important}
</style></head><body>
<header><div class="topline"><div class="eyebrow">Delimit3D / Representation atlas</div><button id="theme" type="button">Dark canvas</button></div>
<h1 id="scene-title"></h1><p>Explore a shared scene through geometry, features and model predictions. Every view uses the same sampled points and a synchronized camera.</p><div class="badges" id="badges"></div></header>
<main><div class="toolbar" aria-label="Viewer controls">
<button id="reset" type="button">Reset view</button><button id="top" type="button">Top view</button>
<label class="control">Point size <input id="size" type="range" min="1" max="6" step=".25" value="2"><output id="size-value">2</output></label>
<label class="control">Height <input id="height" type="range" min="0" max="100" value="100"><output id="height-value">100%</output></label>
<label class="control"><input id="compare" type="checkbox" checked>Compare</label>
<label class="control"><input id="queries" type="checkbox" checked>Query / click markers</label>
<label class="control">Model / checkpoint <select id="model-filter" aria-label="Model or checkpoint"><option value="">All models</option></select></label>
<label class="control">Interaction step <input id="interaction-step" type="range" min="0" max="0" value="0"><output id="step-value">—</output></label>
<label class="control">Failure / diagnostic <select id="failure-filter" aria-label="Failure type"><option value="">All diagnostics</option></select></label>
<label class="control">Object <select id="object-filter" aria-label="Inspect object"><option value="">All objects</option></select></label>
<button id="save-camera" type="button">Save camera JSON</button>
<button id="export-view" type="button">Save current view</button>
</div><div class="workspace" id="workspace"></div>
<div class="inspect"><section class="card"><h2>Point inspector</h2><p id="point-info">Hover a point to preview its original point ID. Click to pin it across both panels.</p></section>
<section class="card"><h2>Reading this scene</h2><p id="source"></p><div class="downloads" id="downloads"></div></section></div>
<section class="library"><h2>Visualization layers</h2><p>Select a layer below to inspect it in the left panel. Use either dropdown to build a comparison.</p><div class="panel-grid" id="panel-grid"></div></section>
<details><summary>Experiment record and figure notes</summary><pre id="record"></pre></details>
<footer>Drag to orbit · scroll to zoom · Shift + drag to pan · click to pin a point · arrows orbit a focused canvas · synchronized cameras. Clicking inspects stored values; it does not recompute model predictions. Coordinates use the dataset's original frame; Z is the height axis. Visual diagnostics support evaluation; they do not establish model quality.</footer>
</main><div id="tooltip"></div><script id="scene-data" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('scene-data').textContent);
const $=id=>document.getElementById(id), state={yaw:D.camera.yaw,pitch:D.camera.pitch,zoom:1,panX:0,panY:0,size:2,height:100,query:true,compare:true,pin:null};
state.objectFilter=null;
state.size=D.camera.point_size||2;
state.height=D.camera.clip_z==null?100:Math.max(0,Math.min(100,100*(D.camera.clip_z-D.z_range[0])/(D.z_range[1]-D.z_range[0]||1)));
$('size').value=state.size;$('size-value').textContent=state.size;
$('height').value=state.height;$('height-value').textContent=state.height.toFixed(1)+'%';
const views=[], rgb=c=>'rgb('+c.join(',')+')';
const fmt=v=>Number(v).toLocaleString(undefined,{maximumFractionDigits:5});
$('scene-title').textContent=D.scene_id;
for(const label of [D.points.length.toLocaleString()+' / '+D.total_points.toLocaleString()+' points',D.panels.length+' visualization layers','Offline · no network required','Sampling seed '+D.seed]){const el=document.createElement('span');el.className='badge';el.textContent=label;$('badges').append(el)}
if(D.metadata.illustrative_only){const el=document.createElement('span');el.className='badge warning';el.textContent='Illustrative '+(D.metadata.synthetic?'synthetic example':'scene')+' · not benchmark evidence';$('badges').append(el)}
const source=D.metadata.source||D.metadata.source_path||D.metadata.path||D.metadata.dataset||'See the experiment record for dataset and model details.';
$('source').textContent=String(source)+' Shared scene framing; scalar limits are stored per diagnostic. Gold stars mark queries; numbered blue circles are positive/object clicks and magenta crosses are negative/background clicks.';
$('record').textContent=JSON.stringify({scene:D.scene_id,point_count:D.total_points,displayed_points:D.points.length,seed:D.seed,scene_metadata:D.metadata,panels:D.panels.map(p=>({key:p.key,title:p.title,caption:p.caption,query:p.query&&p.query.index,legend:p.legend&&{limits:p.legend.limits,label:p.legend.label},metadata:p.metadata}))},null,2);
for(const format of ['png','pdf']){const files=D.files['contact_sheets_'+format]||[];files.forEach((name,i)=>{const a=document.createElement('a');a.href=encodeURIComponent(name);a.textContent='Presentation '+(files.length>1?i+1+' ':'')+format.toUpperCase();a.download=name;$('downloads').append(a)})}
function addView(slot,index){
 const article=document.createElement('article');article.className='view';
 const head=document.createElement('div');head.className='view-head';const span=document.createElement('span');span.className='slot';span.textContent=slot;
 const select=document.createElement('select');select.setAttribute('aria-label','Layer in panel '+slot);
 D.panels.forEach((p,i)=>{const option=document.createElement('option');option.value=i;option.textContent=p.title;select.append(option)});select.value=index;head.append(span,select);
 const stage=document.createElement('div');stage.className='stage';const canvas=document.createElement('canvas');canvas.tabIndex=0;canvas.setAttribute('aria-label','Interactive point cloud panel '+slot);const hint=document.createElement('span');hint.className='hint';hint.textContent='SHARED CAMERA / Z UP';stage.append(canvas,hint);
 const info=document.createElement('div');info.className='panel-info';const caption=document.createElement('p');caption.className='caption';const legend=document.createElement('div');legend.className='legend';const links=document.createElement('div');links.className='downloads';info.append(caption,legend,links);article.append(head,stage,info);$('workspace').append(article);
 const view={article,canvas,select,caption,legend,links,index,projected:[],ctx:canvas.getContext('2d')};views.push(view);
 select.addEventListener('change',()=>{view.index=Number(select.value);updateInfo(view);requestDraw()});bindCanvas(view);updateInfo(view);
}
function updateInfo(view){
 const panel=D.panels[view.index];view.caption.textContent=panel.caption||'Point-aligned visualization in the original scene frame.';view.legend.replaceChildren();
 if(panel.legend){const title=document.createElement('div');title.textContent=panel.legend.label;const gradient=document.createElement('div');gradient.className='gradient';gradient.style.background='linear-gradient(to right,'+panel.legend.colors.map(rgb).join(',')+')';const ticks=document.createElement('div');ticks.className='ticks';for(const value of panel.legend.limits){const el=document.createElement('span');el.textContent=fmt(value);ticks.append(el)}view.legend.append(title,gradient,ticks)}
 else {const categories=panel.metadata.category_legend||panel.metadata.region_legend;if(categories){for(const item of Object.values(categories)){const el=document.createElement('div'),swatch=document.createElement('span');swatch.style.cssText='display:inline-block;width:10px;height:10px;margin-right:6px;border:1px solid #555';swatch.style.background=rgb(item.rgb.map(v=>Math.round(v*255)));el.append(swatch,document.createTextNode(item.label));view.legend.append(el)}}else{const el=document.createElement('span');el.textContent=panel.metadata.legend||'Colors identify appearance or categories; compare matching diagnostics.';view.legend.append(el)}}
 view.links.replaceChildren();for(const format of ['png','pdf']){if(D.files[format]){const name=D.stems[view.index]+'.'+format;const a=document.createElement('a');a.href=encodeURIComponent(name);a.textContent='Paper figure '+format.toUpperCase();a.download=name;view.links.append(a)}}
 document.querySelectorAll('.panel-chip').forEach((el,i)=>el.classList.toggle('active',views.some(v=>v.index===i)));
 if(state.pin!==null)$('point-info').textContent=pointText(state.pin);
}
function worldProjection(point){const x=point[0]-D.center[0],y=point[1]-D.center[1],z=point[2]-D.center[2],cy=Math.cos(state.yaw),sy=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch),away=sy*x+cy*y;return[cy*x-sy*y,cp*z-sp*away,cp*away+sp*z]}
let drawPending=false;
function requestDraw(){if(!drawPending){drawPending=true;requestAnimationFrame(()=>{drawPending=false;drawAll()})}}
function drawAll(){for(const view of views){if(!state.compare&&view===views[1])continue;drawView(view)}}
function star(ctx,x,y,r){ctx.beginPath();for(let i=0;i<10;i++){const angle=-Math.PI/2+i*Math.PI/5,rad=i%2?r*.45:r;const px=x+Math.cos(angle)*rad,py=y+Math.sin(angle)*rad;i?ctx.lineTo(px,py):ctx.moveTo(px,py)}ctx.closePath();ctx.fillStyle='#ffcd55';ctx.strokeStyle='#172b40';ctx.lineWidth=1.5;ctx.fill();ctx.stroke()}
function drawView(view){
 const canvas=view.canvas,ctx=view.ctx,width=canvas.clientWidth,height=canvas.clientHeight,dpr=Math.min(window.devicePixelRatio||1,2);
 if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr)}ctx.setTransform(dpr,0,0,dpr,0,0);
 const dark=document.body.classList.contains('dark');ctx.fillStyle=dark?'#122031':'#f5f7fa';ctx.fillRect(0,0,width,height);
 const scale=Math.min(width,height)*.455*state.zoom/D.radius,ox=width/2+state.panX*width,oy=height/2+state.panY*height,panel=D.panels[view.index],zMax=D.z_range[0]+(D.z_range[1]-D.z_range[0])*state.height/100;
 const projected=[];D.points.forEach((point,i)=>{if(point[2]>zMax+1e-10)return;const p=worldProjection(point),x=ox+p[0]*scale,y=oy-p[1]*scale;if(x>=-8&&x<=width+8&&y>=-8&&y<=height+8)projected.push({x,y,z:p[2],i})});projected.sort((a,b)=>a.z-b.z);view.projected=projected;
 const s=state.size;for(const p of projected){ctx.fillStyle=state.objectFilter!==null&&D.labels&&D.labels[p.i]!==state.objectFilter?(dark?"#344254":"#d3dae0"):rgb(panel.colors[p.i]);ctx.fillRect(p.x-s/2,p.y-s/2,s,s)}
 if(state.query&&panel.query&&panel.query.point[2]<=zMax+1e-10){const p=worldProjection(panel.query.point);star(ctx,ox+p[0]*scale,oy-p[1]*scale,11)}
 if(state.query){for(const click of panel.clicks){if(click.point[2]>zMax+1e-10)continue;const p=worldProjection(click.point),x=ox+p[0]*scale,y=oy-p[1]*scale;const negative=['negative','background'].includes(click.polarity);ctx.beginPath();if(negative){ctx.moveTo(x-7,y-7);ctx.lineTo(x+7,y+7);ctx.moveTo(x+7,y-7);ctx.lineTo(x-7,y+7);ctx.strokeStyle='#CC79A7';ctx.lineWidth=4;ctx.stroke()}else{ctx.arc(x,y,9,0,Math.PI*2);ctx.fillStyle='#0072B2';ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke()}ctx.fillStyle='#fff';ctx.font='bold 10px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(click.ordinal),x,y);ctx.textAlign='left';ctx.textBaseline='alphabetic'}}
 if(state.pin!==null){const p=projected.find(p=>p.i===state.pin);if(p){ctx.beginPath();ctx.arc(p.x,p.y,8,0,Math.PI*2);ctx.strokeStyle=dark?'#fff':'#172b40';ctx.lineWidth=2;ctx.stroke();ctx.beginPath();ctx.arc(p.x,p.y,10,0,Math.PI*2);ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.stroke()}}
 ctx.fillStyle=dark?'#a2b5c6':'#576d80';ctx.font='10px system-ui';ctx.textAlign='right';ctx.fillText(projected.length.toLocaleString()+' visible',width-14,20);ctx.textAlign='left';
}
function nearest(view,event){const rect=view.canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;let best=null,bestDistance=Math.max(7,state.size*2)**2;for(let k=view.projected.length-1;k>=0;k--){const p=view.projected[k],distance=(p.x-x)**2+(p.y-y)**2;if(distance<bestDistance){best=p;bestDistance=distance}}return best}
function pointText(index){const rows=['Original point ID: '+D.indices[index],'XYZ: '+D.points[index].map(fmt).join(', ')];if(D.labels)rows.push('Instance label: '+D.labels[index]);if(D.semantic_labels)rows.push('Semantic label: '+D.semantic_labels[index]);for(const view of views){if(view===views[1]&&!state.compare)continue;const p=D.panels[view.index];if(p.values&&p.values[index]!==null)rows.push(p.title+': '+fmt(p.values[index]))}return rows.join('\n')}
function bindCanvas(view){
 let drag=null;
 view.canvas.addEventListener('pointerdown',event=>{drag={x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,moved:false};view.canvas.setPointerCapture(event.pointerId);$('tooltip').style.display='none'});
 view.canvas.addEventListener('pointermove',event=>{if(drag){const dx=event.clientX-drag.x,dy=event.clientY-drag.y;drag.moved=drag.moved||Math.hypot(event.clientX-drag.startX,event.clientY-drag.startY)>4;if(event.shiftKey){state.panX+=dx/view.canvas.clientWidth;state.panY+=dy/view.canvas.clientHeight}else{state.yaw+=dx*.008;state.pitch=Math.max(-Math.PI/2,Math.min(Math.PI/2,state.pitch+dy*.008))}drag.x=event.clientX;drag.y=event.clientY;requestDraw();return}const point=nearest(view,event);if(point){$('tooltip').textContent='Point '+D.indices[point.i];$('tooltip').style.display='block';$('tooltip').style.left=Math.min(event.clientX+12,window.innerWidth-160)+'px';$('tooltip').style.top=(event.clientY+15)+'px';if(state.pin===null)$('point-info').textContent=pointText(point.i)}else $('tooltip').style.display='none'});
 view.canvas.addEventListener('pointerup',event=>{if(drag&&!drag.moved){const point=nearest(view,event);state.pin=point?point.i:null;$('point-info').textContent=point?pointText(point.i):'Selection cleared. Hover or click a point to inspect it.';requestDraw()}drag=null});
 view.canvas.addEventListener('pointercancel',()=>drag=null);view.canvas.addEventListener('pointerleave',()=>$('tooltip').style.display='none');
 view.canvas.addEventListener('wheel',event=>{event.preventDefault();state.zoom=Math.max(.2,Math.min(12,state.zoom*Math.exp(-event.deltaY*.001)));requestDraw()},{passive:false});
 view.canvas.addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)){event.preventDefault();state.yaw+=(event.key==='ArrowLeft'?-.1:event.key==='ArrowRight'?.1:0);state.pitch=Math.max(-Math.PI/2,Math.min(Math.PI/2,state.pitch+(event.key==='ArrowUp'?.1:event.key==='ArrowDown'?-.1:0)));requestDraw()}});
}
addView('A',0);addView('B',Math.min(1,D.panels.length-1));
D.panels.forEach((panel,i)=>{const button=document.createElement('button');button.className='panel-chip';button.type='button';const title=document.createElement('span');title.textContent=panel.title;const subtitle=document.createElement('small');subtitle.textContent=panel.key;button.append(title,subtitle);button.addEventListener('click',()=>{views[0].index=i;views[0].select.value=i;updateInfo(views[0]);requestDraw()});$('panel-grid').append(button)});views.forEach(updateInfo);
$('size').addEventListener('input',event=>{state.size=Number(event.target.value);$('size-value').textContent=state.size;requestDraw()});
$('height').addEventListener('input',event=>{state.height=Number(event.target.value);$('height-value').textContent=state.height+'%';requestDraw()});
$('queries').addEventListener('change',event=>{state.query=event.target.checked;requestDraw()});
$('compare').addEventListener('change',event=>{state.compare=event.target.checked;views[1].article.classList.toggle('hidden',!state.compare);$('workspace').classList.toggle('single',!state.compare);requestDraw()});
$('reset').addEventListener('click',()=>{state.yaw=D.camera.yaw;state.pitch=D.camera.pitch;state.zoom=1;state.panX=0;state.panY=0;requestDraw()});
$('top').addEventListener('click',()=>{state.pitch=Math.PI/2;state.yaw=0;requestDraw()});
$('theme').addEventListener('click',()=>{const dark=document.body.classList.toggle('dark');$('theme').textContent=dark?'Light canvas':'Dark canvas';requestDraw()});
$('export-view').addEventListener('click',async()=>{
 const selected=state.compare?views:[views[0]],scale=2,width=selected.reduce((sum,v)=>sum+v.canvas.clientWidth,0),height=Math.max(...selected.map(v=>v.canvas.clientHeight))+130;
 const out=document.createElement('canvas');out.width=width*scale;out.height=height*scale;
 const ctx=out.getContext('2d');ctx.scale(scale,scale);ctx.fillStyle='#fff';ctx.fillRect(0,0,width,height);let x=0;
 for(const view of selected){
  const w=view.canvas.clientWidth,panel=D.panels[view.index],y=48+view.canvas.clientHeight;
  ctx.fillStyle='#172b40';ctx.font='bold 13px system-ui';ctx.fillText(panel.title,x+16,26,w-30);ctx.drawImage(view.canvas,x,40,w,view.canvas.clientHeight);
  if(panel.legend){const legend=panel.legend,gradient=ctx.createLinearGradient(x+22,0,x+w-22,0);legend.colors.forEach((c,i)=>gradient.addColorStop(i/Math.max(1,legend.colors.length-1),'rgb('+c.join(',')+')'));ctx.fillStyle=gradient;ctx.fillRect(x+22,y,w-44,7);ctx.fillStyle='#172b40';ctx.font='10px system-ui';ctx.fillText(String(legend.limits[0]),x+22,y+22);ctx.textAlign='right';ctx.fillText(String(legend.limits[1]),x+w-22,y+22);ctx.textAlign='center';ctx.fillText(legend.label,x+w/2,y+36,w-60);ctx.textAlign='left'}
  x+=w;
 }
 ctx.font='10px system-ui';ctx.fillStyle='#526075';ctx.fillText('Exploratory capture · '+D.scene_id+' · camera, numeric scales and source hashes in matching JSON',16,height-15,width-30);
 const hashBytes=async value=>crypto.subtle?Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',value))).map(v=>v.toString(16).padStart(2,'0')).join(''):null;
 const sha=value=>hashBytes(new TextEncoder().encode(value));const pngData=out.toDataURL('image/png');
 const record={schema:'delimit3d_point_view_capture/v1',kind:'exploratory_saved_artifact_view',scene_id:D.scene_id,
  camera:{projection:'orthographic',up_axis:'z',center:D.center,radius:D.radius,...state,clip_z:D.z_range[0]+(D.z_range[1]-D.z_range[0])*state.height/100},
  object_filter:$('object-filter').value,original_point_ids:D.indices,source_metadata:D.metadata,source_report:new URL('report.json',location.href).href,
  available_panel_records:D.panels.map(({key,metadata})=>({key,metadata})),
  payload_sha256:await sha($('scene-data').textContent),renderer_sha256:await sha(document.scripts[document.scripts.length-1].textContent),
  panels:selected.map(v=>{const {colors,values,...p}=D.panels[v.index];return p}),image:{width:out.width,height:out.height,sha256:await hashBytes(Uint8Array.from(atob(pngData.split(',')[1]),c=>c.charCodeAt(0)))}};
 const png=document.createElement('a');png.download='delimit3d-current-view.png';png.href=pngData;png.click();
 const url=URL.createObjectURL(new Blob([JSON.stringify(record,null,2)],{type:'application/json'}));const sidecar=document.createElement('a');sidecar.download='delimit3d-current-view.json';sidecar.href=url;sidecar.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
if('ResizeObserver'in window){const observer=new ResizeObserver(requestDraw);views.forEach(v=>observer.observe(v.canvas))}else window.addEventListener('resize',requestDraw);
requestDraw();

const models=[...new Set(D.panels.map(p=>p.metadata.model_name).filter(Boolean))];
models.forEach(name=>{const o=document.createElement('option');o.value=name;o.textContent=name;$('model-filter').append(o)});
const failures=[...new Set(D.panels.map(p=>p.metadata.failure_type).filter(Boolean))];failures.forEach(name=>{const o=document.createElement('option');o.value=name;o.textContent=name.replaceAll('_',' ');$('failure-filter').append(o)});$('failure-filter').disabled=!failures.length;
const steps=D.panels.filter(p=>Number.isInteger(p.metadata.step)).map(p=>p.metadata.step);
$('interaction-step').max=steps.length?Math.max(...steps):0;$('interaction-step').disabled=!steps.length;
function chooseResearchPanel(){
 const name=$('model-filter').value,step=Number($('interaction-step').value),failure=$('failure-filter').value;
 const scoped=D.panels.map((p,i)=>({p,i})).filter(x=>(!name||x.p.metadata.model_name===name)&&(!failure||x.p.metadata.failure_type===failure));
 const temporal=scoped.filter(x=>Number.isInteger(x.p.metadata.step));
 const matches=temporal.length?temporal.filter(x=>x.p.metadata.step===step):scoped;
 const preferred=matches.find(x=>!x.p.metadata.failure_type)||matches[0];
 if(preferred){const select=views[0].select;select.value=preferred.i;select.dispatchEvent(new Event('change'));}
 $('step-value').textContent=steps.length?String(step):'—';
 document.querySelectorAll('.panel-chip').forEach((el,i)=>{el.hidden=Boolean(name&&D.panels[i].metadata.model_name!==name)});
}
$('failure-filter').addEventListener('change',chooseResearchPanel);$('model-filter').addEventListener('change',chooseResearchPanel);$('interaction-step').addEventListener('input',()=>{chooseResearchPanel();const step=Number($('interaction-step').value);for(const view of views.slice(1)){const current=D.panels[view.index],name=current.metadata.model_name;if(!name)continue;const i=D.panels.findIndex(p=>p.metadata.model_name===name&&p.metadata.step===step&&!p.metadata.failure_type);if(i>=0){view.select.value=i;view.select.dispatchEvent(new Event('change'))}}});
if(D.labels){[...new Set(D.labels)].filter(x=>x>=0).sort((a,b)=>a-b).forEach(id=>{const o=document.createElement('option');o.value=id;o.textContent='GT '+id;$('object-filter').append(o)})}
$('object-filter').addEventListener('change',()=>{const val=$('object-filter').value;if(val===''){state.objectFilter=null;state.pin=null;drawAll();return;}state.objectFilter=Number(val);const id=Number(val),indices=D.labels.map((v,i)=>v===id?i:-1).filter(i=>i>=0);if(indices.length){state.pin=indices[0];$('point-info').textContent=pointText(indices[0]);drawAll();}});
$('save-camera').addEventListener('click',()=>{const data={schema:'delimit3d_view_camera/v1',scene_id:D.scene_id,projection:'orthographic',up_axis:'z',center:D.center,yaw:state.yaw,pitch:state.pitch,zoom:state.zoom,panX:state.panX,panY:state.panY,point_size:state.size,height_percent:state.height};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='camera.json';a.click();URL.revokeObjectURL(url)});
</script></body></html>'''

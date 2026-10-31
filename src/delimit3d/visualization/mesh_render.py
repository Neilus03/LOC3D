"""Complete triangle surfaces for offline exploration and publication figures.

The browser uses WebGL depth testing and interpolated vertex colors. Static
exports use depth-sorted triangles with mean vertex colors and no illumination;
the distinction is recorded rather than silently changing diagnostic palettes.
Neither renderer drops faces or randomly subsamples a mesh.
"""

from __future__ import annotations

import base64
import html
import math
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .render import (
    _PITCH, _YAW, _click_markers, _compact_metadata, _legend, _matplotlib,
    _project, _rgb, _safe_stem, _script_json,
)

if TYPE_CHECKING:
    from .contracts import Panel, Scene


def _binary(array: np.ndarray, dtype: str) -> dict[str, Any]:
    """JSON-safe little-endian typed arrays without decimal-coordinate bloat."""
    packed = np.ascontiguousarray(array, dtype=np.dtype(dtype).newbyteorder("<"))
    return {"dtype": packed.dtype.str, "shape": list(packed.shape),
            "base64": base64.b64encode(packed.tobytes()).decode("ascii")}


def _categorical(panel: Panel) -> bool:
    """Suite RGB/PCA colors interpolate; discrete labels retain palette entries."""
    explicit = panel.metadata.get("categorical")
    if explicit is not None:
        return bool(explicit)
    return (panel.values is None and panel.key != "rgb" and
            not panel.key.endswith("/pca"))


def _face_colors(vertex_colors: np.ndarray, faces: np.ndarray, *, categorical: bool
                 ) -> np.ndarray:
    colors = vertex_colors[faces]
    if not categorical:
        return colors.mean(axis=1)
    # A majority vote prevents invented labels. If all three disagree, use the
    # first vertex in the supplied face ordering; topology is left unchanged.
    first = colors[:, 0].copy()
    last_agree = np.all(colors[:, 1] == colors[:, 2], axis=1)
    first[last_agree] = colors[last_agree, 1]
    return first


def _mesh_payload(scene: Scene, panels: list[Panel], seed: int) -> dict[str, Any]:
    mesh = scene.mesh
    points = np.asarray(scene.points, dtype=np.float64)
    indices = np.asarray(mesh.point_indices, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    # Use scene-wide framing, as the point renderer does, even when a mesh uses
    # only some points. Include reconstructed vertices if they extend beyond it.
    lower = np.minimum(points.min(axis=0), vertices.min(axis=0))
    upper = np.maximum(points.max(axis=0), vertices.max(axis=0))
    center = np.asarray(scene.metadata.get("camera", {}).get("center", (lower + upper) / 2))
    radius = max(float(np.linalg.norm(points - center, axis=1).max()),
                 float(np.linalg.norm(vertices - center, axis=1).max()), 1e-6)
    packed_panels = []
    for panel in panels:
        colors = _rgb(panel.colors)
        values = None
        if panel.values is not None:
            array = np.asarray(panel.values, dtype=np.float64).reshape(-1)[indices].copy()
            undefined = panel.metadata.get("undefined_mask")
            if undefined is not None:
                mask = np.asarray(undefined, dtype=bool)
                if mask.shape != (len(points),):
                    raise ValueError(f"Panel {panel.key!r}: undefined_mask must align with scene points")
                array[mask[indices]] = np.nan
            values = _binary(array, "f8")
        query = None if panel.query_index is None else {
            "index": int(panel.query_index),
            "point": points[int(panel.query_index)].tolist(),
        }
        packed_panels.append({
            "key": panel.key, "title": panel.title, "caption": panel.caption,
            "colors": _binary(colors[indices], "u1"), "values": values,
            "legend": _legend(panel, colors), "query": query,
            "clicks": _click_markers(scene, panel),
            "categorical": _categorical(panel),
            "metadata": _compact_metadata(panel.metadata),
        })
    return {
        "scene_id": scene.scene_id, "vertices": _binary(vertices, "f4"),
        "faces": _binary(mesh.faces, "u4"),
        "point_indices": _binary(indices, "u4"),
        "labels": None if scene.labels is None else _binary(np.asarray(scene.labels)[indices], "f8"),
        "semantic_labels": (None if scene.semantic_labels is None else
                            _binary(np.asarray(scene.semantic_labels)[indices], "f8")),
        "total_points": len(points), "vertex_count": len(vertices),
        "face_count": len(mesh.faces), "center": center.tolist(), "radius": radius,
        "z_range": [float(lower[2]), float(upper[2])],
        "camera": scene.metadata.get("camera", {"yaw": _YAW, "pitch": _PITCH}), "seed": seed,
        "metadata": _compact_metadata(scene.metadata),
        "mesh_metadata": _compact_metadata(mesh.metadata),
        "rendering": {"topology": "All supplied triangles; no display simplification",
                      "html_colors": "Continuous: interpolated vertex RGB; categorical: majority vertex color (first vertex on ties); unlit",
                      "static_colors": "Continuous: mean triangle vertex RGB; categorical: majority vertex color (first vertex on ties); unlit",
                      "static_visibility": "Depth-sorted triangles (painter's algorithm)",
                      "markers": "Original scene point positions; drawn over the surface"},
        "panels": packed_panels,
    }


def _mesh_note(scene: Scene, payload: dict[str, Any]) -> str:
    metadata = scene.mesh.metadata
    method = str(metadata.get("reconstruction", metadata.get(
        "method", metadata.get("kind", "supplied triangle surface"))))
    source = str(metadata.get("source_kind", metadata.get("source_type", "")))
    description = f"{source}: {method}" if source and source != method else method
    if metadata.get("reconstructed") or "reconstruct" in description.lower():
        description = "Reconstructed / approximate surface · " + description
    note = ("Illustrative example; not benchmark evidence. "
            if scene.metadata.get("illustrative_only") else "")
    return (note + f"{description} · {payload['vertex_count']:,} vertices / "
            f"{payload['face_count']:,} triangles · full supplied topology · "
            "unlit RGB (continuous: triangle mean; categories: majority, first vertex on ties) · "
            "shared camera · star = query; circles = clicks")


def _static_exports(scene: Scene, panels: list[Panel], output_dir: Path,
                    formats: tuple[str, ...], stems: list[str],
                    payload: dict[str, Any]) -> dict[str, Any]:
    plt, LinearSegmentedColormap, Normalize, ScalarMappable = _matplotlib()
    from matplotlib.collections import PolyCollection

    center = np.asarray(payload["center"])
    projected = _project(np.asarray(scene.mesh.vertices), center, payload["camera"])
    scene_projection = _project(np.asarray(scene.points), center, payload["camera"])
    lo = np.minimum(projected[:, :2].min(axis=0), scene_projection[:, :2].min(axis=0))
    hi = np.maximum(projected[:, :2].max(axis=0), scene_projection[:, :2].max(axis=0))
    span = hi - lo
    margin = np.maximum(span, max(float(span.max()), 1e-6) * .08) * .06
    limits = np.column_stack((lo - margin, hi + margin))
    faces = np.asarray(scene.mesh.faces, dtype=np.int64)
    clip_z = payload["camera"].get("clip_z")
    visible_faces = np.arange(len(faces))
    if clip_z is not None:
        visible_faces = visible_faces[np.all(scene.mesh.vertices[faces,2] <= clip_z, axis=1)]
        payload["rendering"]["static_clip"] = {"clip_z":clip_z, "policy":"whole triangle; every vertex below plane", "visible_faces":len(visible_faces), "note":"Boundary triangles omitted in static view; original topology unchanged"}
    order = visible_faces[np.argsort(projected[faces[visible_faces], 2].mean(axis=1), kind="stable")]
    triangles = projected[faces[order], :2]
    indices = np.asarray(scene.mesh.point_indices, dtype=np.int64)
    result = {fmt: [] for fmt in formats}

    def draw(ax, panel, *, title_size=12, show_caption=True):
        vertex_colors = _rgb(panel.colors)[indices].astype(np.float32) / 255.0
        face_colors = _face_colors(vertex_colors, faces[order], categorical=_categorical(panel))
        collection = PolyCollection(triangles, facecolors=face_colors,
                                    edgecolors="none", linewidths=0,
                                    antialiaseds=False, rasterized=True)
        ax.add_collection(collection)
        if panel.query_index is not None and (clip_z is None or scene.points[panel.query_index,2] <= clip_z):
            point = _project(np.asarray(scene.points)[[int(panel.query_index)]], center, payload["camera"])[0]
            ax.scatter([point[0]], [point[1]], s=110, marker="*", c="#ffcc50",
                       edgecolors="#142233", linewidths=.9, zorder=10)
        for click in _click_markers(scene, panel):
            if clip_z is not None and click["point"][2] > clip_z:continue
            point = _project(np.asarray([click["point"]]), center, payload["camera"])[0]
            color = ("#bd4646" if click["polarity"] in {"negative", "background"} else
                     "#208866" if click["polarity"] == "positive" else "#326da8")
            ax.scatter([point[0]], [point[1]], s=95, marker="x" if click["polarity"] in {"negative", "background"} else "o", c=color,
                       edgecolors="white", linewidths=1, zorder=11)
            ax.text(point[0], point[1], str(click["ordinal"]), color="white",
                    fontsize=6, ha="center", va="center", zorder=12)
        ax.set(xlim=tuple(limits[0]), ylim=tuple(limits[1]), aspect="equal")
        ax.axis("off")
        ax.set_title(textwrap.fill(panel.title, 42), fontsize=title_size,
                     fontweight="semibold", color="#192b41", pad=8)
        if show_caption and panel.caption:
            ax.text(.5, -.22, textwrap.fill(panel.caption, 80), ha="center", va="top",
                    transform=ax.transAxes, fontsize=7, color="#526075")
        legend = _legend(panel, _rgb(panel.colors))
        if legend is not None:
            lower, upper = legend["limits"]
            if upper > lower:
                cmap = LinearSegmentedColormap.from_list(
                    "mesh_panel_palette", np.asarray(legend["colors"]) / 255.)
                scalar = ScalarMappable(norm=Normalize(vmin=lower, vmax=upper), cmap=cmap)
                cax = ax.inset_axes([.18, -.105 if show_caption else -.02, .64, .021])
                bar = ax.figure.colorbar(scalar, cax=cax, orientation="horizontal")
                bar.set_ticks([lower, (lower + upper) / 2, upper])
                bar.ax.tick_params(labelsize=6, length=2, pad=1)
                bar.outline.set_visible(False)
                bar.set_label(str(legend["label"]), fontsize=6, labelpad=1)

    note = _mesh_note(scene, payload)
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.facecolor": "white"}):
        for panel, stem in zip(panels, stems):
            fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="white")
            fig.subplots_adjust(left=.07, right=.93, top=.86, bottom=.30)
            draw(ax, panel)
            fig.text(.075, .95, "DELIMIT3D / SURFACE ATLAS", fontsize=7,
                     color="#537080", fontweight="bold")
            fig.text(.075, .055, textwrap.fill(f"{scene.scene_id} · {note}", 110),
                     fontsize=6.5, color="#526075")
            for fmt in formats:
                filename = f"{stem}.{fmt}"
                fig.savefig(output_dir / filename, dpi=240, bbox_inches="tight")
                result[fmt].append(filename)
            plt.close(fig)
        pages = max(1, math.ceil(len(panels) / 6))
        contact_files = {fmt: [] for fmt in formats}
        for page in range(pages):
            subset = panels[page * 6:(page + 1) * 6]
            columns = min(3, len(subset))
            rows = math.ceil(len(subset) / columns)
            fig, axes = plt.subplots(rows, columns, figsize=(16, 9), squeeze=False,
                                     facecolor="white")
            fig.subplots_adjust(left=.035, right=.97, top=.85, bottom=.13,
                                wspace=.1, hspace=.38)
            for ax, panel in zip(axes.flat, subset):
                draw(ax, panel, title_size=11, show_caption=False)
            for ax in list(axes.flat)[len(subset):]:
                ax.set_visible(False)
            fig.text(.045, .945, "DELIMIT3D / SURFACE ATLAS", fontsize=10,
                     color="#537080", fontweight="bold")
            fig.text(.045, .89, str(scene.scene_id), fontsize=20, color="#192b41")
            fig.text(.045, .035, textwrap.fill(note, 155), fontsize=8, color="#526075")
            fig.text(.96, .935, f"{page + 1:02d} / {pages:02d}", ha="right",
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


def render_mesh_scene(scene: Scene, panels: list[Panel], output_dir: Path, *,
                      formats: tuple[str, ...] = ("html", "png", "pdf"),
                      seed: int = 0) -> dict[str, Any]:
    """Render a scene mesh with exactly the suite's original point-aligned colors.

    Vertex correspondence is ``panel.colors[scene.mesh.point_indices]``. All
    supplied triangles survive export, regardless of point-gallery sampling.
    Returned paths are relative to ``output_dir``.
    """
    formats = tuple(dict.fromkeys(formats))
    unknown = set(formats) - {"html", "png", "pdf"}
    if unknown:
        raise ValueError(f"Unsupported mesh render formats: {', '.join(sorted(unknown))}")
    if getattr(scene, "mesh", None) is None:
        raise ValueError("Mesh rendering requires scene.mesh")
    scene.validate()
    if not panels:
        raise ValueError("At least one visualization panel is required")
    stems = [_safe_stem(panel.key) for panel in panels]
    if len(set(stems)) != len(stems) or any(stem.startswith("contact_sheet") for stem in stems):
        raise ValueError("Panel keys must have unique, non-reserved filename forms")
    for panel in panels:
        if np.asarray(panel.colors).shape != (len(scene.points), 3):
            raise ValueError(f"Panel {panel.key!r} must have one RGB color per scene point")
        if panel.values is not None and np.asarray(panel.values).size != len(scene.points):
            raise ValueError(f"Panel {panel.key!r} values do not align with scene points")
        if panel.query_index is not None and (
            not isinstance(panel.query_index, (int, np.integer))
            or not 0 <= int(panel.query_index) < len(scene.points)
        ):
            raise ValueError(f"Panel {panel.key!r} query index is outside the scene")
        if panel.limits is not None and (
            len(panel.limits) != 2 or not np.isfinite(panel.limits).all()
            or panel.limits[0] > panel.limits[1]
        ):
            raise ValueError(f"Panel {panel.key!r} requires finite ordered scalar limits")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _mesh_payload(scene, panels, seed)
    result: dict[str, Any] = {}
    static = tuple(fmt for fmt in formats if fmt in {"png", "pdf"})
    if static:
        result.update(_static_exports(scene, panels, output_dir, static, stems, payload))
    if "html" in formats:
        payload["files"], payload["stems"] = result, stems
        payload["surface_note"] = _mesh_note(scene, payload)
        document = _HTML.replace("__TITLE__", html.escape(str(scene.scene_id)))
        document = document.replace("__PAYLOAD__", _script_json(payload))
        (output_dir / "index.html").write_text(document, encoding="utf-8")
        result["html"] = "index.html"
    return result


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · Delimit3D surface atlas</title><style>
:root{color-scheme:light;--bg:#edf1f4;--surface:#fff;--ink:#172b40;--muted:#576d80;--line:#dce4ea;--accent:#147e89;--canvas:#f5f7fa}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}body.dark{color-scheme:dark;--bg:#101b28;--surface:#192738;--ink:#e3eef7;--muted:#a2b5c6;--line:#304253;--accent:#67d3d1;--canvas:#122031}button,select,input{font:inherit}button,select{background:var(--surface);color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:7px 10px}button{cursor:pointer}button:hover{border-color:var(--accent)}button:focus-visible,select:focus-visible,input:focus-visible,canvas:focus-visible{outline:2px solid var(--accent);outline-offset:2px}a{color:var(--accent)}header{max-width:1560px;padding:31px 32px 20px;margin:auto}.eyebrow{text-transform:uppercase;letter-spacing:.19em;font-size:10px;font-weight:750;color:var(--accent)}h1{font-size:clamp(25px,3vw,37px);letter-spacing:-.035em;line-height:1.15;margin:12px 0 9px;overflow-wrap:anywhere}header p{max-width:960px;color:var(--muted);margin:0}.topline{display:flex;justify-content:space-between;align-items:center;gap:12px}.badges{display:flex;gap:8px;flex-wrap:wrap;margin-top:17px}.badge{font-size:11px;color:var(--muted);background:var(--surface);border:1px solid var(--line);border-radius:30px;padding:5px 11px}.warning{color:#835614;background:#fff2d3;border-color:#efdeb6}.toolbar{display:flex;gap:15px;align-items:center;flex-wrap:wrap;padding:15px 18px;background:var(--surface);border:1px solid var(--line);border-radius:12px}.control{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted)}.control input[type=range]{width:100px;accent-color:var(--accent)}main{max-width:1560px;margin:auto;padding:0 32px 36px}.workspace{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}.workspace.single{grid-template-columns:1fr}.view{background:var(--surface);border:1px solid var(--line);border-radius:13px;overflow:hidden}.view-head{padding:15px 17px 13px;display:flex;align-items:center;justify-content:space-between;gap:12px}.view-head select{max-width:calc(100% - 44px);font-weight:600;font-size:13px}.slot{color:var(--accent);font-size:11px;font-weight:750;letter-spacing:.1em}.stage{position:relative;background:var(--canvas);height:440px}.stage canvas{position:absolute;inset:0;display:block;width:100%;height:100%}.stage .overlay{cursor:grab;touch-action:none}.stage .overlay:active{cursor:grabbing}.hint{position:absolute;left:15px;bottom:12px;color:var(--muted);font-size:10px;pointer-events:none}.panel-info{min-height:112px;padding:12px 17px 16px}.caption{font-size:12px;color:var(--muted);margin:0 0 10px;min-height:18px;overflow-wrap:anywhere}.legend{width:235px;max-width:100%;font-size:10px;color:var(--muted)}.gradient{height:8px;border-radius:3px;margin:4px 0}.ticks{display:flex;justify-content:space-between}.downloads{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0 2px;font-size:11px}.inspect{margin-top:18px;display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{border:1px solid var(--line);background:var(--surface);border-radius:12px;padding:18px;min-width:0}.card h2{font-size:12px;margin:0 0 10px;font-weight:700}.card p{font-size:12px;color:var(--muted);margin:0;overflow-wrap:anywhere}#point-info{font-variant-numeric:tabular-nums;white-space:pre-wrap}details{margin-top:18px;font-size:12px;color:var(--muted)}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;border:1px solid var(--line);padding:14px;border-radius:8px;max-height:300px;overflow:auto;font-size:10px}footer{font-size:11px;color:var(--muted);margin-top:28px}.hidden{display:none!important}#status{font-size:11px;margin:10px 2px;color:var(--muted)}.error{color:#bd4646!important}@media(max-width:900px){header{padding:23px 18px 18px}main{padding:0 18px 25px}.workspace,.inspect{grid-template-columns:1fr}.stage{height:380px}.toolbar{gap:11px}.view-head select{font-size:12px}}@media print{.toolbar,.hint,details,#theme{display:none}body{background:#fff}.workspace{grid-template-columns:1fr 1fr}.stage{height:330px}.view,.card{break-inside:avoid}}
</style></head><body>
<header><div class="topline"><div class="eyebrow">Delimit3D / Surface atlas</div><button id="theme" type="button">Dark canvas</button></div><h1 id="scene-title"></h1>
<p>Explore diagnostics on complete triangle surfaces. Linked cameras and the original diagnostic palettes make feature and prediction comparisons easy to read.</p><div class="badges" id="badges"></div></header>
<main><div class="toolbar" aria-label="Mesh viewer controls">
<button id="reset" type="button">Reset view</button><button id="top" type="button">Top view</button>
<label class="control">Surface <select id="mode"><option value="surface">Mesh</option><option value="edges">Mesh + edges</option><option value="wireframe">Wireframe</option></select></label>
<label class="control">Height <input id="height" type="range" min="0" max="100" value="100"><output id="height-value">100%</output></label>
<label class="control"><input id="compare" type="checkbox" checked>Compare</label>
<label class="control"><input id="queries" type="checkbox" checked>Query / click markers</label>
<button id="export-view" type="button">Save view + record</button></div>
<p id="status" role="status">Loading complete mesh…</p><div class="workspace" id="workspace"></div>
<div class="inspect"><section class="card"><h2>Surface inspector</h2><p id="point-info">Click a surface to inspect the nearest vertex of its frontmost triangle and the associated original point ID.</p></section>
<section class="card"><h2>Reading this surface</h2><p id="source"></p><div class="downloads" id="downloads"></div></section></div>
<details><summary>Experiment record and figure notes</summary><pre id="record"></pre></details>
<footer>Drag to orbit · scroll to zoom · Shift + drag to pan · arrows orbit a focused canvas · synchronized cameras · Z is the height axis. Gold stars mark queries; numbered circles mark recorded clicks. Markers overlay the surface. The inspector reports stored values; it does not recompute model predictions. Continuous browser colors interpolate vertex RGB; static figures use triangle-mean RGB. Categories use the majority vertex color of each face (first vertex on ties), so no extra categories are invented. No illumination changes diagnostic colors. Static exports use depth sorting; intersecting triangles can differ from WebGL visibility.</footer>
</main><script id="scene-data" type="application/json">__PAYLOAD__</script><script>
'use strict';
const D=JSON.parse(document.getElementById('scene-data').textContent),$=id=>document.getElementById(id);
function unpack(a){if(!a)return null;const raw=atob(a.base64),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);const types={'<f4':Float32Array,'<f8':Float64Array,'<u4':Uint32Array,'|u1':Uint8Array};return new types[a.dtype](bytes.buffer)}
const P=unpack(D.vertices),F=unpack(D.faces),I=unpack(D.point_indices),labels=unpack(D.labels),semantic=unpack(D.semantic_labels);
for(const p of D.panels){p.colors=unpack(p.colors);p.values=unpack(p.values)}
const state={yaw:D.camera.yaw,pitch:D.camera.pitch,zoom:1,panX:0,panY:0,height:D.camera.clip_z==null?100:Math.max(0,Math.min(100,100*(D.camera.clip_z-D.z_range[0])/(D.z_range[1]-D.z_range[0]||1))),query:true,compare:true,mode:'surface',pin:null};$('height').value=state.height;$('height-value').textContent=state.height.toFixed(1)+'%';
const views=[],fmt=v=>Number(v).toLocaleString(undefined,{maximumFractionDigits:6}),rgb=c=>'rgb('+c.join(',')+')';
$('scene-title').textContent=D.scene_id;
for(const label of [D.vertex_count.toLocaleString()+' vertices',D.face_count.toLocaleString()+' triangles · full topology',D.panels.length+' layers','Offline · no network required','Unlit · original palettes']){const el=document.createElement('span');el.className='badge';el.textContent=label;$('badges').append(el)}
if(D.metadata.illustrative_only){const el=document.createElement('span');el.className='badge warning';el.textContent='Illustrative example · not benchmark evidence';$('badges').append(el)}
$('source').textContent=D.surface_note+'; positive clicks green, negative/background red, other clicks blue.';
$('record').textContent=JSON.stringify({scene:D.scene_id,original_points:D.total_points,vertices:D.vertex_count,triangles:D.face_count,mesh:D.mesh_metadata,rendering:D.rendering,scene_metadata:D.metadata,seed:D.seed,panels:D.panels.map(p=>({key:p.key,title:p.title,caption:p.caption,legend:p.legend,query:p.query,clicks:p.clicks,metadata:p.metadata}))},null,2);
for(const format of ['png','pdf'])for(const [i,name] of (D.files['contact_sheets_'+format]||[]).entries()){const a=document.createElement('a');a.href=encodeURIComponent(name);a.download=name;a.textContent='Presentation '+(i+1)+' '+format.toUpperCase();$('downloads').append(a)}
const vertexSource=`attribute vec3 aPosition;attribute vec3 aColor;uniform vec3 uCenter;uniform vec4 uRotation;uniform vec2 uScale;uniform vec2 uPan;uniform float uRadius;varying vec3 vColor;varying float vHeight;void main(){vec3 p=aPosition-uCenter;float h=uRotation.x*p.x-uRotation.y*p.y;float away=uRotation.y*p.x+uRotation.x*p.y;float up=uRotation.z*p.z-uRotation.w*away;float depth=uRotation.z*away+uRotation.w*p.z;gl_Position=vec4(h*uScale.x+uPan.x,up*uScale.y+uPan.y,-depth/(uRadius*2.0),1.0);vColor=aColor;vHeight=aPosition.z;}`;
const fragmentSource=`precision mediump float;varying vec3 vColor;varying float vHeight;uniform float uHeight;uniform float uWire;void main(){if(vHeight>uHeight)discard;gl_FragColor=vec4(mix(vColor,vec3(0.13,0.22,0.29),uWire),1.0);}`;
function program(gl){const p=gl.createProgram();for(const [kind,source] of [[gl.VERTEX_SHADER,vertexSource],[gl.FRAGMENT_SHADER,fragmentSource]]){const shader=gl.createShader(kind);gl.shaderSource(shader,source);gl.compileShader(shader);if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw Error('Shader compilation failed: '+gl.getShaderInfoLog(shader));gl.attachShader(p,shader)}gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error('Shader link failed: '+gl.getProgramInfoLog(p));return p}
function buffer(gl,target,data){const b=gl.createBuffer();gl.bindBuffer(target,b);gl.bufferData(target,data,gl.STATIC_DRAW);return b}
function initGL(view){const gl=view.canvas.getContext('webgl',{antialias:true,alpha:false,preserveDrawingBuffer:true});if(!gl)throw Error('WebGL is unavailable. Enable browser hardware graphics, or open the exported PNG/PDF figures.');view.gl=gl;view.program=program(gl);gl.useProgram(view.program);view.locations={};for(const name of ['uCenter','uRotation','uScale','uPan','uRadius','uHeight','uWire'])view.locations[name]=gl.getUniformLocation(view.program,name);view.pos=gl.getAttribLocation(view.program,'aPosition');view.color=gl.getAttribLocation(view.program,'aColor');view.uint=Boolean(gl.getExtension('OES_element_index_uint'));view.count=F.length;
 if(view.uint){view.positions=buffer(gl,gl.ARRAY_BUFFER,P);view.indices=buffer(gl,gl.ELEMENT_ARRAY_BUFFER,F)}else{const expanded=new Float32Array(F.length*3);for(let i=0;i<F.length;i++)expanded.set(P.subarray(F[i]*3,F[i]*3+3),i*3);view.positions=buffer(gl,gl.ARRAY_BUFFER,expanded)}
 view.colors=gl.createBuffer();view.colorIndex=-1;view.edges=null;gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.disable(gl.CULL_FACE);view.canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();$('status').textContent='Graphics context lost. Reload to restore the mesh; exported figures remain available.';$('status').classList.add('error')})}
function setColors(view){if(view.colorIndex===view.index)return;const gl=view.gl,panel=D.panels[view.index],original=panel.colors;view.expanded=!view.uint||panel.categorical;if(panel.categorical&&view.uint&&!view.expandedPositions){const expanded=new Float32Array(F.length*3);for(let i=0;i<F.length;i++)expanded.set(P.subarray(F[i]*3,F[i]*3+3),i*3);view.expandedPositions=buffer(gl,gl.ARRAY_BUFFER,expanded)}let colors=original;if(view.expanded){colors=new Uint8Array(F.length*3);if(panel.categorical){const equal=(a,b)=>original[a]===original[b]&&original[a+1]===original[b+1]&&original[a+2]===original[b+2];for(let i=0;i<F.length;i+=3){const a=F[i]*3,b=F[i+1]*3,c=F[i+2]*3,chosen=equal(b,c)?b:a;for(let j=0;j<3;j++)colors.set(original.subarray(chosen,chosen+3),(i+j)*3)}}else for(let i=0;i<F.length;i++)colors.set(original.subarray(F[i]*3,F[i]*3+3),i*3)}gl.bindBuffer(gl.ARRAY_BUFFER,view.colors);gl.bufferData(gl.ARRAY_BUFFER,colors,gl.STATIC_DRAW);view.colorIndex=view.index}
function setEdges(view){if(view.edges)return;const gl=view.gl;const edges=new Uint32Array(F.length*2);for(let i=0;i<F.length;i+=3){edges.set([F[i],F[i+1],F[i+1],F[i+2],F[i+2],F[i]],i*2)}if(view.uint){view.edges=buffer(gl,gl.ELEMENT_ARRAY_BUFFER,edges)}else{const positions=new Float32Array(edges.length*3);for(let i=0;i<edges.length;i++)positions.set(P.subarray(edges[i]*3,edges[i]*3+3),i*3);view.edges=buffer(gl,gl.ARRAY_BUFFER,positions)}}
function addView(slot,index){const article=document.createElement('article');article.className='view';const head=document.createElement('div');head.className='view-head';const label=document.createElement('span');label.className='slot';label.textContent=slot;const select=document.createElement('select');select.setAttribute('aria-label','Layer in panel '+slot);D.panels.forEach((p,i)=>{const option=document.createElement('option');option.value=i;option.textContent=p.title;select.append(option)});select.value=index;head.append(label,select);const stage=document.createElement('div');stage.className='stage';const canvas=document.createElement('canvas'),overlay=document.createElement('canvas');overlay.className='overlay';overlay.tabIndex=0;overlay.setAttribute('aria-label','Interactive mesh panel '+slot);const hint=document.createElement('span');hint.className='hint';hint.textContent='FULL TOPOLOGY / SHARED CAMERA / Z UP';stage.append(canvas,overlay,hint);const info=document.createElement('div');info.className='panel-info';const caption=document.createElement('p');caption.className='caption';const legend=document.createElement('div');legend.className='legend';const links=document.createElement('div');links.className='downloads';info.append(caption,legend,links);article.append(head,stage,info);$('workspace').append(article);const view={article,canvas,overlay,select,caption,legend,links,index,ctx:overlay.getContext('2d')};views.push(view);try{initGL(view)}catch(error){view.error=error.message;$('status').textContent=error.message;$('status').classList.add('error')}select.addEventListener('change',()=>{view.index=Number(select.value);updateInfo(view);requestDraw()});bindCanvas(view);updateInfo(view)}
function updateInfo(view){const p=D.panels[view.index];view.caption.textContent=p.caption||'Point-aligned colors transferred through the recorded vertex mapping.';view.legend.replaceChildren();if(p.legend){const title=document.createElement('div');title.textContent=p.legend.label;const gradient=document.createElement('div');gradient.className='gradient';gradient.style.background='linear-gradient(to right,'+p.legend.colors.map(rgb).join(',')+')';const ticks=document.createElement('div');ticks.className='ticks';for(const value of p.legend.limits){const el=document.createElement('span');el.textContent=fmt(value);ticks.append(el)}view.legend.append(title,gradient,ticks)}else{view.legend.textContent=p.metadata.legend||'Appearance / categorical colors'}view.links.replaceChildren();for(const format of ['png','pdf'])if(D.files[format]){const name=D.stems[view.index]+'.'+format;const a=document.createElement('a');a.href=encodeURIComponent(name);a.download=name;a.textContent='Paper '+format.toUpperCase();view.links.append(a)}const ply=(D.mesh_metadata.ply_exports||[])[view.index];if(ply){const a=document.createElement('a');a.href=encodeURIComponent(ply);a.download=ply;a.textContent='Colored mesh PLY';view.links.append(a)}if(state.pin!==null)$('point-info').textContent=pointText(state.pin)}
function project(point){const x=point[0]-D.center[0],y=point[1]-D.center[1],z=point[2]-D.center[2],cy=Math.cos(state.yaw),sy=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch),away=sy*x+cy*y;return[cy*x-sy*y,cp*z-sp*away,cp*away+sp*z]}
function vertex(i){return P.subarray(i*3,i*3+3)}
let pending=false;function requestDraw(){if(!pending){pending=true;requestAnimationFrame(()=>{pending=false;drawAll()})}}
function drawAll(){for(const view of views)if(state.compare||view===views[0])drawView(view)}
function star(ctx,x,y){ctx.beginPath();for(let i=0;i<10;i++){const a=-Math.PI/2+i*Math.PI/5,r=i%2?5:11;i?ctx.lineTo(x+Math.cos(a)*r,y+Math.sin(a)*r):ctx.moveTo(x+Math.cos(a)*r,y+Math.sin(a)*r)}ctx.closePath();ctx.fillStyle='#ffcd55';ctx.strokeStyle='#172b40';ctx.lineWidth=1.5;ctx.fill();ctx.stroke()}
function drawView(view){const width=view.overlay.clientWidth,height=view.overlay.clientHeight,dpr=Math.min(window.devicePixelRatio||1,2);if(!width||!height)return;for(const canvas of [view.canvas,view.overlay])if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr)}const ctx=view.ctx;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);const dark=document.body.classList.contains('dark'),scale=Math.min(width,height)*.455*state.zoom/D.radius,zMax=D.z_range[0]+(D.z_range[1]-D.z_range[0])*state.height/100,ox=width/2+state.panX*width,oy=height/2+state.panY*height;
 if(!view.error){const gl=view.gl;gl.viewport(0,0,view.canvas.width,view.canvas.height);gl.clearColor(...(dark?[18/255,32/255,49/255,1]:[245/255,247/255,250/255,1]));gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(view.program);setColors(view);const u=view.locations;gl.uniform3fv(u.uCenter,D.center);gl.uniform4f(u.uRotation,Math.cos(state.yaw),Math.sin(state.yaw),Math.cos(state.pitch),Math.sin(state.pitch));gl.uniform2f(u.uScale,2*scale/width,2*scale/height);gl.uniform2f(u.uPan,2*state.panX,-2*state.panY);gl.uniform1f(u.uRadius,D.radius);gl.uniform1f(u.uHeight,zMax+1e-5);gl.uniform1f(u.uWire,0);gl.bindBuffer(gl.ARRAY_BUFFER,view.expanded&&view.uint?view.expandedPositions:view.positions);gl.enableVertexAttribArray(view.pos);gl.vertexAttribPointer(view.pos,3,gl.FLOAT,false,0,0);gl.bindBuffer(gl.ARRAY_BUFFER,view.colors);gl.enableVertexAttribArray(view.color);gl.vertexAttribPointer(view.color,3,gl.UNSIGNED_BYTE,true,0,0);
 if(state.mode!=='wireframe'){gl.enable(gl.POLYGON_OFFSET_FILL);gl.polygonOffset(1,1);if(!view.expanded){gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,view.indices);gl.drawElements(gl.TRIANGLES,view.count,gl.UNSIGNED_INT,0)}else gl.drawArrays(gl.TRIANGLES,0,view.count);gl.disable(gl.POLYGON_OFFSET_FILL)}
 if(state.mode!=='surface'){setEdges(view);gl.uniform1f(u.uWire,1);if(view.uint){gl.bindBuffer(gl.ARRAY_BUFFER,view.positions);gl.vertexAttribPointer(view.pos,3,gl.FLOAT,false,0,0);gl.disableVertexAttribArray(view.color);gl.vertexAttrib3f(view.color,0,0,0);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,view.edges);gl.drawElements(gl.LINES,F.length*2,gl.UNSIGNED_INT,0)}else{gl.disableVertexAttribArray(view.color);gl.vertexAttrib3f(view.color,0,0,0);gl.bindBuffer(gl.ARRAY_BUFFER,view.edges);gl.vertexAttribPointer(view.pos,3,gl.FLOAT,false,0,0);gl.drawArrays(gl.LINES,0,F.length*2)}}
 }else{ctx.fillStyle='#bd4646';ctx.font='12px system-ui';ctx.fillText('WebGL unavailable. Use the exported figures.',16,35)}
 const panel=D.panels[view.index];if(state.query&&panel.query&&panel.query.point[2]<=zMax+1e-5){const p=project(panel.query.point);star(ctx,ox+p[0]*scale,oy-p[1]*scale)}if(state.query)for(const click of panel.clicks){if(click.point[2]>zMax+1e-5)continue;const p=project(click.point),x=ox+p[0]*scale,y=oy-p[1]*scale;ctx.beginPath();ctx.arc(x,y,9,0,2*Math.PI);ctx.fillStyle=['negative','background'].includes(click.polarity)?'#bd4646':click.polarity==='positive'?'#208866':'#326da8';ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();ctx.fillStyle='#fff';ctx.font='bold 10px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(click.ordinal),x,y);ctx.textAlign='left';ctx.textBaseline='alphabetic'}if(state.pin!==null&&vertex(state.pin)[2]<=zMax+1e-5){const p=project(vertex(state.pin));ctx.beginPath();ctx.arc(ox+p[0]*scale,oy-p[1]*scale,9,0,2*Math.PI);ctx.strokeStyle=dark?'#fff':'#172b40';ctx.lineWidth=3;ctx.stroke();ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.stroke()}ctx.fillStyle=dark?'#a2b5c6':'#576d80';ctx.font='10px system-ui';ctx.textAlign='right';ctx.fillText(D.face_count.toLocaleString()+' triangles'+(state.height<100?' · clipped by height':''),width-14,20);ctx.textAlign='left';view.drawCount=(view.drawCount||0)+1}
function pick(view,event){const rect=view.overlay.getBoundingClientRect(),scale=Math.min(rect.width,rect.height)*.455*state.zoom/D.radius,x=(event.clientX-rect.left-rect.width/2-state.panX*rect.width)/scale,y=-(event.clientY-rect.top-rect.height/2-state.panY*rect.height)/scale,zMax=D.z_range[0]+(D.z_range[1]-D.z_range[0])*state.height/100;const projected=new Float32Array(P.length);for(let i=0;i<D.vertex_count;i++)projected.set(project(vertex(i)),i*3);let best=null,depth=-Infinity;for(let i=0;i<F.length;i+=3){const a=F[i]*3,b=F[i+1]*3,c=F[i+2]*3,ax=projected[a],ay=projected[a+1],bx=projected[b],by=projected[b+1],cx=projected[c],cy=projected[c+1],den=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy);if(Math.abs(den)<1e-14)continue;const u=((by-cy)*(x-cx)+(cx-bx)*(y-cy))/den,v=((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/den,w=1-u-v;if(u<0||v<0||w<0)continue;const worldZ=u*P[a+2]+v*P[b+2]+w*P[c+2],z=u*projected[a+2]+v*projected[b+2]+w*projected[c+2];if(worldZ>zMax+1e-5||z<=depth)continue;depth=z;let distance=Infinity;for(const j of [a,b,c]){const d=(projected[j]-x)**2+(projected[j+1]-y)**2;if(d<distance){best=j/3;distance=d}}}return best}
function pointText(i){const rows=['Mesh vertex: '+i,'Original point ID: '+I[i],'Mesh XYZ: '+Array.from(vertex(i)).map(fmt).join(', ')];if(labels)rows.push('Instance label: '+labels[i]);if(semantic)rows.push('Semantic label: '+semantic[i]);for(const view of views){if(!state.compare&&view===views[1])continue;const p=D.panels[view.index];if(p.values)rows.push(p.title+': '+(Number.isFinite(p.values[i])?fmt(p.values[i]):'undefined'))}return rows.join('\n')}
function bindCanvas(view){let drag=null;const canvas=view.overlay;canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,sx:e.clientX,sy:e.clientY,moved:false};canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.moved=drag.moved||Math.hypot(e.clientX-drag.sx,e.clientY-drag.sy)>4;if(e.shiftKey){state.panX+=dx/canvas.clientWidth;state.panY+=dy/canvas.clientHeight}else{state.yaw+=dx*.008;state.pitch=Math.max(-Math.PI/2,Math.min(Math.PI/2,state.pitch+dy*.008))}drag.x=e.clientX;drag.y=e.clientY;requestDraw()});canvas.addEventListener('pointerup',e=>{if(drag&&!drag.moved){state.pin=pick(view,e);$('point-info').textContent=state.pin===null?'No surface selected. Click a visible triangle.':pointText(state.pin);requestDraw()}drag=null});canvas.addEventListener('pointercancel',()=>drag=null);canvas.addEventListener('wheel',e=>{e.preventDefault();state.zoom=Math.max(.2,Math.min(12,state.zoom*Math.exp(-e.deltaY*.001)));requestDraw()},{passive:false});canvas.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)){e.preventDefault();state.yaw+=(e.key==='ArrowLeft'?-.1:e.key==='ArrowRight'?.1:0);state.pitch=Math.max(-Math.PI/2,Math.min(Math.PI/2,state.pitch+(e.key==='ArrowUp'?.1:e.key==='ArrowDown'?-.1:0)));requestDraw()}})}
addView('A',0);addView('B',Math.min(1,D.panels.length-1));
if(views.every(v=>!v.error))$('status').textContent='WebGL depth-tested surfaces · '+(views.every(v=>v.uint)?'indexed triangles':'complete non-indexed compatibility mode')+' · no mesh simplification';
$('height').addEventListener('input',e=>{state.height=Number(e.target.value);$('height-value').textContent=state.height+'%';requestDraw()});$('mode').addEventListener('change',e=>{state.mode=e.target.value;requestDraw()});$('queries').addEventListener('change',e=>{state.query=e.target.checked;requestDraw()});$('compare').addEventListener('change',e=>{state.compare=e.target.checked;views[1].article.classList.toggle('hidden',!state.compare);$('workspace').classList.toggle('single',!state.compare);requestDraw()});$('reset').addEventListener('click',()=>{state.yaw=D.camera.yaw;state.pitch=D.camera.pitch;state.zoom=1;state.panX=0;state.panY=0;requestDraw()});$('top').addEventListener('click',()=>{state.pitch=Math.PI/2;state.yaw=0;requestDraw()});$('theme').addEventListener('click',()=>{const dark=document.body.classList.toggle('dark');$('theme').textContent=dark?'Light canvas':'Dark canvas';requestDraw()});
async function captureBytesHash(bytes){const digest=await crypto.subtle.digest('SHA-256',bytes);return Array.from(new Uint8Array(digest),v=>v.toString(16).padStart(2,'0')).join('')}
function captureTextHash(text){return captureBytesHash(new TextEncoder().encode(text))}
function captureDownload(name,url){const a=document.createElement('a');a.download=name;a.href=url;a.click()}
function captureLegend(ctx,panel,x,y,width){ctx.fillStyle='#526075';ctx.font='12px system-ui';if(!panel.legend){ctx.fillText('Non-scalar colors; panel interpretation in JSON',x,y,width);return}const legend=panel.legend,barWidth=Math.min(260,width),gradient=ctx.createLinearGradient(x,0,x+barWidth,0);legend.colors.forEach((color,i)=>gradient.addColorStop(i/Math.max(1,legend.colors.length-1),rgb(color)));ctx.fillText(legend.label,x,y,width);ctx.fillStyle=gradient;ctx.fillRect(x,y+10,barWidth,9);ctx.fillStyle='#526075';ctx.font='11px system-ui';ctx.textAlign='left';ctx.fillText(fmt(legend.limits[0]),x,y+35);ctx.textAlign='right';ctx.fillText(fmt(legend.limits[legend.limits.length-1]),x+barWidth,y+35);ctx.textAlign='left'}
$('export-view').addEventListener('click',async()=>{
 const button=$('export-view');button.disabled=true;
 try{
  drawAll();const chosen=state.compare?views:[views[0]],scale=2,width=chosen.reduce((sum,v)=>sum+v.canvas.clientWidth,0),stageHeight=Math.max(...chosen.map(v=>v.canvas.clientHeight)),height=stageHeight+174;
  const record={schema:'delimit3d.mesh_view_capture.v1',scope:'exploratory browser capture; not benchmark evidence',scene_id:D.scene_id,seed:D.seed,
   source:{page_url:location.href,scene_metadata:D.metadata,mesh_metadata:D.mesh_metadata,original_point_count:D.total_points,vertex_count:D.vertex_count,face_count:D.face_count,available_panel_source_records:D.panels.filter(p=>p.metadata.source_record).map(p=>({key:p.key,model_name:p.metadata.model_name,feature_level:p.metadata.feature_level,source_record:p.metadata.source_record}))},
   camera:{projection:'orthographic',up_axis:'z',center:D.center,radius:D.radius,framing_factor:0.455,yaw:state.yaw,pitch:state.pitch,zoom:state.zoom,pan_fraction:{x:state.panX,y:state.panY},height_percent:state.height,z_range:D.z_range,clip_z:D.z_range[0]+(D.z_range[1]-D.z_range[0])*state.height/100,clip_comparison_epsilon:1e-5},
   interactive_state:{...state},initial_camera:D.camera,rendering:{...D.rendering,backend:'browser WebGL depth test',mode:state.mode,theme:document.body.classList.contains('dark')?'dark':'light',markers_visible:state.query,inspected_mesh_vertex:state.pin,inspected_original_point_id:state.pin===null?null:Number(I[state.pin]),capture_scale:scale,width_pixels:width*scale,height_pixels:height*scale},
   panels:chosen.map(view=>{const p=D.panels[view.index];return{index:view.index,key:p.key,title:p.title,caption:p.caption,legend:p.legend,categorical:p.categorical,query:p.query,clicks:p.clicks,metadata:p.metadata,viewport_css_pixels:[view.canvas.clientWidth,view.canvas.clientHeight],viewport_buffer_pixels:[view.canvas.width,view.canvas.height]}}),
   files:{image:'delimit3d-mesh-current-view.png',record:'delimit3d-mesh-current-view.json'}};
  const out=document.createElement('canvas');out.width=width*scale;out.height=height*scale;const ctx=out.getContext('2d');ctx.scale(scale,scale);ctx.fillStyle='#fff';ctx.fillRect(0,0,width,height);let x=0;
  for(const view of chosen){const w=view.canvas.clientWidth,panel=D.panels[view.index];ctx.fillStyle='#172b40';ctx.font='bold 13px system-ui';ctx.fillText(panel.title,x+16,26,w-30);ctx.drawImage(view.canvas,x,40,w,view.canvas.clientHeight);ctx.drawImage(view.overlay,x,40,w,view.overlay.clientHeight);captureLegend(ctx,panel,x+16,stageHeight+62,w-32);x+=w}
  ctx.font='10px system-ui';ctx.fillStyle='#526075';ctx.fillText(D.scene_id+' · full mesh / unlit vertex RGB / height '+state.height+'%',16,height-42,width-32);ctx.fillText('Exploratory capture · companion JSON records camera, scales and sources · not benchmark evidence',16,height-20,width-32);
  const imageUrl=out.toDataURL('image/png');
  // Hash the original embedded payload text, never the decoded/mutated arrays.
  const imageRaw=atob(imageUrl.split(',')[1]),imageBytes=new Uint8Array(imageRaw.length);for(let i=0;i<imageRaw.length;i++)imageBytes[i]=imageRaw.charCodeAt(i);
  const hashes=await Promise.all([captureTextHash($('scene-data').textContent),captureTextHash(document.scripts[document.scripts.length-1].textContent),captureBytesHash(imageBytes)]);record.source.payload_sha256=hashes[0];record.source.renderer_script_sha256=hashes[1];record.files.image_sha256=hashes[2];
  const recordUrl=URL.createObjectURL(new Blob([JSON.stringify(record,null,2)+'\n'],{type:'application/json'}));
  captureDownload(record.files.image,imageUrl);captureDownload(record.files.record,recordUrl);setTimeout(()=>URL.revokeObjectURL(recordUrl),60000);
  $('status').textContent='Saved exploratory PNG and companion JSON with current camera, scales and source records.';
 }catch(error){$('status').textContent='View export failed: '+error.message;$('status').classList.add('error')}
 finally{button.disabled=false}
});
if('ResizeObserver'in window){const observer=new ResizeObserver(requestDraw);views.forEach(v=>observer.observe(v.overlay))}else window.addEventListener('resize',requestDraw);requestDraw();
</script></body></html>'''

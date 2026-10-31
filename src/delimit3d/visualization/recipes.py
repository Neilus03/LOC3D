"""Explicit figure composition, cameras and context-preserving close-up recipes."""
from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path
import math
import textwrap

import numpy as np

from .contracts import Scene, Panel


def camera_record(scene: Scene, config: dict | None = None) -> dict:
    config = dict(config or {})
    camera = {"yaw": -.68, "pitch": .62, "projection": "orthographic", "up_axis": "z",
              "point_size": 1.5, "clip_z": None,
              "center": ((scene.points.min(0) + scene.points.max(0)) / 2).tolist()}
    camera.update(config)
    if camera["projection"] != "orthographic" or camera["up_axis"] != "z":
        raise ValueError("Only orthographic Z-up cameras are supported")
    if (np.asarray(camera["center"]).shape != (3,) or
            not np.isfinite([*camera["center"], camera["yaw"], camera["pitch"], camera["point_size"]]).all() or
            camera["point_size"] <= 0):
        raise ValueError("Camera values must be finite and point size positive")
    if camera["clip_z"] is not None and not np.isfinite(camera["clip_z"]):
        raise ValueError("clip_z must be finite")
    return camera


def project(points: np.ndarray, camera: dict) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float64) - camera["center"]
    cy, sy = math.cos(camera["yaw"]), math.sin(camera["yaw"])
    cp, sp = math.cos(camera["pitch"]), math.sin(camera["pitch"])
    horizontal = cy * xyz[:, 0] - sy * xyz[:, 1]
    away = sy * xyz[:, 0] + cy * xyz[:, 1]
    return np.column_stack((horizontal, cp * xyz[:, 2] - sp * away, cp * away + sp * xyz[:, 2]))


def select_panels(panels: list[Panel], selectors: list[str]) -> list[Panel]:
    selected = []
    for selector in selectors:
        matches = [p for p in panels if fnmatchcase(p.key, selector)]
        if not matches:
            raise FileNotFoundError(f"Recipe panel is unavailable: {selector}")
        selected.extend(p for p in matches if p.key not in {item.key for item in selected})
    return selected


def crop_record(scene: Scene, config: dict | None, camera: dict) -> dict | None:
    if not config:
        return None
    config = dict(config)
    padding = float(config.get("padding", .25))
    if padding < 0 or not np.isfinite(padding):
        raise ValueError("Crop padding must be finite and nonnegative")
    if "bounds" in config:
        bounds = np.asarray(config["bounds"], dtype=float)
    elif "target_id" in config:
        if scene.labels is None or not np.any(scene.labels == config["target_id"]):
            raise ValueError("Crop target_id is absent from scene ground truth")
        points = scene.points[scene.labels == config["target_id"]]
        bounds = np.stack([points.min(0) - padding, points.max(0) + padding])
    elif "query_index" in config:
        q = config["query_index"]
        if not isinstance(q, int) or not 0 <= q < len(scene.points):
            raise ValueError("Crop query_index must be an original point index")
        radius = float(config.get("radius", 1.))
        if radius <= 0 or not np.isfinite(radius):
            raise ValueError("Crop radius must be positive")
        bounds = np.stack([scene.points[q] - radius, scene.points[q] + radius])
    else:
        raise ValueError("Crop requires bounds, target_id or query_index")
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError("Crop bounds must be finite ordered [minXYZ,maxXYZ]")
    corners = np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
    projected = project(corners, camera)
    return {**config, "bounds": bounds.tolist(), "padding": padding,
            "projected_bounds": [projected[:,:2].min(0).tolist(),projected[:,:2].max(0).tolist()],
            "selection": "explicit world-space crop; camera shared with full context"}


def render_recipes(scene: Scene, panels: list[Panel], output: Path, *, recipes: list[dict],
                   sample_indices: np.ndarray, camera: dict, formats=("png", "pdf"),
                   source_record: dict | None = None) -> dict:
    """Render named panel grids at exact paper and presentation dimensions.

    A crop always adds a full-scene context row using the identical panels/camera.
    Geometry is never changed; original mesh topology remains in the mesh gallery.
    These figures use sampled points and explicitly record their original IDs.
    """
    from .render import _matplotlib, _legend, _rgb, _click_markers, _safe_stem
    from .suite import write_json, array_hash
    plt, LinearSegmentedColormap, Normalize, ScalarMappable = _matplotlib()
    from matplotlib.patches import Rectangle
    output.mkdir(exist_ok=True)
    indices = np.asarray(sample_indices)
    records = []
    names = set()
    for recipe in recipes:
        name = _safe_stem(recipe["name"])
        if name in names:
            raise ValueError("Recipe names must have unique filename forms")
        names.add(name)
        try:
            chosen = select_panels(panels, recipe["panels"])
        except FileNotFoundError as exc:
            records.append({"name": name, "status": "blocked", "reason": str(exc)})
            continue
        if not chosen:
            raise ValueError("Recipe must contain at least one panel")
        crop = crop_record(scene, recipe.get("crop"), camera)
        full = project(scene.points, camera)
        keep = np.ones(len(indices), bool)
        if camera["clip_z"] is not None:
            keep &= scene.points[indices,2] <= camera["clip_z"]
        shown = indices[keep]
        order = np.argsort(full[shown,2], kind="stable"); shown = shown[order]
        np.save(output / f"{name}.sample_indices.npy", shown)
        crop_indices = None
        if crop:
            b=np.asarray(crop['bounds'])
            eligible=np.flatnonzero(np.all((scene.points>=b[0])&(scene.points<=b[1]),axis=1))
            if camera['clip_z'] is not None:eligible=eligible[scene.points[eligible,2]<=camera['clip_z']]
            limit=int(recipe.get('max_crop_points',12000))
            if limit<1:raise ValueError('max_crop_points must be positive')
            sampling_seed=int(recipe.get('sampling_seed',42))
            chosen_ids=eligible if len(eligible)<=limit else np.sort(np.random.default_rng(sampling_seed).choice(eligible,limit,replace=False))
            crop_indices=chosen_ids[np.argsort(full[chosen_ids,2],kind='stable')]
            np.save(output/f'{name}.crop_indices.npy',crop_indices)
            crop.update(sample_indices=f'{name}.crop_indices.npy',sample_sha256=array_hash(crop_indices),
                        eligible_points=len(eligible),display_points=len(crop_indices),sampling_seed=sampling_seed,
                        method='all crop points up to max_crop_points, otherwise uniform fixed-seed original IDs')
        marker_policy=recipe.get('click_marker_policy','all')
        if marker_policy not in {'all','latest'}:raise ValueError('click_marker_policy must be all or latest')
        xy = full[shown,:2]
        lo,hi = full[:,:2].min(0),full[:,:2].max(0)
        margin = np.maximum(hi-lo,.01)*.05
        full_bounds = np.stack([lo-margin,hi+margin])
        cols = int(recipe.get("columns", min(4,len(chosen))))
        if cols < 1 or cols > 6:
            raise ValueError("Recipe columns must be between 1 and 6")
        groups = [chosen[i:i+cols] for i in range(0,len(chosen),cols)]
        row_count = len(groups) * (2 if crop else 1)
        record = {"schema":"delimit3d_figure/v1", "name":name,"status":"complete",
                  "scene_id":scene.scene_id,"dataset":scene.metadata.get("dataset"),
                  "recipe":recipe,"camera":camera,"crop":crop,"click_marker_policy":marker_policy,
                  "point_count":len(scene.points),"display_points":len(shown),
                  "sampling":"fixed shared original-point sample; no model inference",
                  "sample_indices":f"{name}.sample_indices.npy","sample_sha256":array_hash(shown),
                  "geometry_sha256":array_hash(scene.points),"source_record":source_record or {},
                  "panels":[{"key":p.key,"title":p.title,"caption":p.caption,"limits":p.limits,
                             "query_index":p.query_index,"metadata":p.metadata} for p in chosen],"files":[],
                  "scale_policy":"Shared only for identical quantity/semantics; panel limits retained",
                  "rendering":"CPU orthographic depth-sorted point splats; original topology retained in mesh outputs"}
        if row_count > 4:
            raise ValueError("Recipe exceeds four readable rows; split it into multiple named recipes")
        domains = sorted({p.metadata["metric_domain"] for p in chosen if p.metadata.get("metric_domain")})
        budgets = sorted({p.metadata["click_budget"]["definition"] for p in chosen if p.metadata.get("click_budget")})
        incomplete = any(p.metadata.get("pairing", {}).get("missing") for p in chosen)
        caption = recipe.get("caption", "Selected-scene diagnostic; see figure JSON for artifacts and measurement domain.")
        mandatory = []
        if domains: mandatory.append("Metric domain: " + ", ".join(domains))
        if budgets: mandatory.append("Click budget: " + "; ".join(budgets))
        if incomplete: mandatory.append("Encoder/decoder/preprocessing record incomplete")
        if any(p.key.endswith("/pca") for p in chosen): mandatory.append("PCA colors do not establish semantic alignment")
        modes = sorted({p.metadata["mode"] for p in chosen if p.metadata.get("mode")})
        if modes: mandatory.append("Mode: " + ", ".join(modes))
        if marker_policy=='latest':mandatory.append('Only the latest physical click marker is shown per frame; full actual sequence is in the gallery and figure record')
        if mandatory: caption += " " + ". ".join(mandatory) + "."
        if scene.metadata.get("synthetic"):
            caption = "SYNTHETIC FIXTURE — not experimental evidence. " + caption
        record["visible_caption"] = caption
        for size in recipe.get("sizes",["paper","presentation"]):
            if size not in {"paper","presentation"}:
                raise ValueError("Recipe sizes are paper or presentation")
            width,dpi = (7.2,300) if size=="paper" else (40/3,144)
            caption_font = 6.3 if size=="paper" else 9
            caption_wrap = 145 if size=="paper" else 160
            caption_lines = textwrap.wrap(caption, caption_wrap)
            footer_inches = .22
            caption_inches = len(caption_lines) * caption_font * 1.25 / 72
            caption_area_inches = max(.8, caption_inches + footer_inches + .25)
            # Colorbars and categorical legends extend below the point axes.
            # Give them their own physical band instead of sharing caption space.
            has_legends=any(_legend(p,_rgb(p.colors)) or p.metadata.get('category_legend') or p.metadata.get('region_legend') for p in chosen)
            decoration_inches = .65 if has_legends else .1
            bottom_inches = caption_area_inches + decoration_inches
            main_font,subtitle_font=(10,6) if size=='paper' else (18,9)
            main_lines=textwrap.wrap(recipe.get('title',scene.scene_id),max(20,int((width-.5)*72/(main_font*.6))))
            subtitle=f"{scene.scene_id} · {scene.metadata.get('experiment','experiment recorded in JSON')} · {len(shown):,}/{len(scene.points):,} context points"
            if crop:subtitle+=f" · {len(crop_indices):,}/{crop['eligible_points']:,} detail points"
            subtitle_lines=textwrap.wrap(subtitle,max(30,int((width-.5)*72/(subtitle_font*.56))))
            title_top_inches=.15 if size=='paper' else .2
            subtitle_top_inches=title_top_inches+len(main_lines)*main_font*1.25/72+.15
            top_inches=subtitle_top_inches+len(subtitle_lines)*subtitle_font*1.25/72+(.65 if size=='paper' else .95)+.1
            height = max(3.8,row_count*1.7 + top_inches + bottom_inches) if size=="paper" else 7.5
            title_size = 7 if size=="paper" else 11
            title_wrap = max(18,int((width-.5)/cols*72/(title_size*.66)))
            with plt.rc_context({"font.family":"DejaVu Sans","pdf.fonttype":42,"savefig.facecolor":"white"}):
                fig, axes=plt.subplots(row_count,cols,figsize=(width,height),squeeze=False)
                bottom_decorations=[]
                fig.subplots_adjust(left=.035,right=.98,bottom=bottom_inches/height,top=1-top_inches/height,wspace=.12,hspace=.65)
                for g, group in enumerate(groups):
                    for mode in range(2 if crop else 1):
                        row = g*(2 if crop else 1)+mode
                        for col in range(cols):
                            ax=axes[row,col]
                            if col>=len(group):ax.axis('off');continue
                            panel=group[col]
                            selected=shown
                            bounds=full_bounds
                            if mode:
                                selected=crop_indices;bounds=np.asarray(crop['projected_bounds'])
                            ax.scatter(full[selected,0],full[selected,1],c=panel.colors[selected],s=float(camera['point_size']),linewidths=0,rasterized=True)
                            if crop and not mode:
                                b=np.asarray(crop['projected_bounds']);ax.add_patch(Rectangle(b[0],*(b[1]-b[0]),fill=False,edgecolor='#333333',linewidth=.7,linestyle='--'))
                            def marker_visible(index):
                                if camera['clip_z'] is not None and scene.points[index,2] > camera['clip_z']:return False
                                if mode:
                                    b=np.asarray(crop['bounds'])
                                    return bool(np.all((scene.points[index]>=b[0]) & (scene.points[index]<=b[1])))
                                return True
                            if panel.query_index is not None and marker_visible(panel.query_index):
                                q=full[panel.query_index];ax.scatter(q[0],q[1],marker='*',s=48,c='#E69F00',edgecolors='#151515',linewidths=.5,zorder=6)
                            markers=_click_markers(scene,panel)
                            if marker_policy=='latest':markers=markers[-1:]
                            for click in markers:
                                if not marker_visible(click['index']):continue
                                q=full[click['index']];negative=click['polarity'] in {'negative','background'}
                                ax.scatter(q[0],q[1],s=14,marker='x' if negative else 'o',c='#CC79A7' if negative else '#0072B2',linewidths=.9,zorder=7)
                                ax.annotate(str(click['ordinal']),q[:2],xytext=(3,2),textcoords='offset points',fontsize=5,color='#111111',zorder=8)
                            ax.set(xlim=bounds[:,0],ylim=bounds[:,1],aspect='equal');ax.axis('off')
                            title=recipe.get('panel_titles',{}).get(panel.key,panel.title)
                            annotations = []
                            for metric in panel.metadata.get('recorded_metrics',[]):
                                if metric.get('target_id') is None or len(panel.metadata.get('targets',[]))==1 or str(metric.get('target_id'))==str(panel.metadata.get('target_id')):
                                    value=metric['value'];unit=metric.get('unit','')
                                    if unit=='fraction':value*=100;unit='%'
                                    annotations.append(f"{metric['name']} {value:.1f}{unit}")
                            proposal=panel.metadata.get('proposal',{})
                            if proposal.get('score') is not None and 'score' not in title.lower():annotations.append(f"score {proposal['score']:.3f}")
                            if panel.metadata.get('score') is not None and 'score' not in title.lower():annotations.append(f"score {panel.metadata['score']:.3f}")
                            if annotations:title += "\n" + ' · '.join(annotations[:2])
                            ax.set_title('\n'.join(textwrap.fill(line,title_wrap) for line in (('Detail · ' if mode else '')+title).splitlines()),fontsize=title_size,pad=5)
                            categories=panel.metadata.get('category_legend',panel.metadata.get('region_legend',{}))
                            if categories and (mode or not crop):
                                for j,item in enumerate(categories.values()):
                                    if j>=12:break
                                    wide=any(len(item['label'])>15 for item in categories.values()) and len(categories)<=4
                                    x=.04+(0 if wide else (j%2)*.49);y=-.05-(j if wide else j//2)*.105
                                    ax.scatter([x],[y],s=14,c=[item['rgb']],transform=ax.transAxes,clip_on=False)
                                    artist=ax.text(x+.035,y,item['label'],transform=ax.transAxes,fontsize=5.5 if size=='paper' else 8,va='center',clip_on=False)
                                    if row==row_count-1:bottom_decorations.append(artist)
                            legend=_legend(panel,_rgb(panel.colors))
                            if legend and (mode or not crop):
                                low,high=legend['limits']
                                if high>low:
                                    cmap=LinearSegmentedColormap.from_list('exact',np.asarray(legend['colors'])/255)
                                    cax=ax.inset_axes([.12,-.14,.76,.035]);bar=fig.colorbar(ScalarMappable(norm=Normalize(low,high),cmap=cmap),cax=cax,orientation='horizontal')
                                    bar.set_ticks([low,0 if low<0<high else (low+high)/2,high]);bar.ax.tick_params(labelsize=6,length=2,pad=1);bar.outline.set_visible(False)
                                    bar.set_label(legend['label'],fontsize=6,labelpad=1)
                                    if row==row_count-1:bottom_decorations.append(cax)
                fig.text(.035,1-title_top_inches/height,'\n'.join(main_lines),fontsize=main_font,weight='semibold',va='top',linespacing=1.25)
                subtitle_artist=fig.text(.035,1-subtitle_top_inches/height,'\n'.join(subtitle_lines),fontsize=subtitle_font,color='#526075',va='top',linespacing=1.25)
                caption_artist=fig.text(.035,(caption_area_inches-.18)/height,'\n'.join(caption_lines),fontsize=caption_font,color='#334155',va='top',linespacing=1.25)
                fig.text(.035,.10/height,'Star: fixed query · circles / crosses: positive / negative clicks · dashed box: fixed world-space crop',fontsize=5.8 if size=='paper' else 8,color='#526075')
                fig.canvas.draw()
                renderer=fig.canvas.get_renderer()
                caption_top=caption_artist.get_window_extent(renderer).y1
                legend_bottom=min((artist.get_tightbbox(renderer).y0 for artist in bottom_decorations),default=caption_top+1)
                if legend_bottom<=caption_top:
                    raise ValueError(f'Recipe {name} {size}: legend overlaps caption; use fewer panels/legend entries')
                highest_panel_title=max(ax.title.get_window_extent(renderer).y1 for ax in axes[0] if ax.get_visible())
                subtitle_bottom=subtitle_artist.get_window_extent(renderer).y0
                if highest_panel_title>=subtitle_bottom:
                    raise ValueError(f'Recipe {name} {size}: panel title overlaps subtitle; shorten panel titles')
                record.setdefault('layout_checks',{})[size]={'legend_caption_overlap':False,
                    'legend_caption_gap_inches':(legend_bottom-caption_top)/fig.dpi if bottom_decorations else None,
                    'panel_title_subtitle_overlap':False,'panel_title_subtitle_gap_inches':(subtitle_bottom-highest_panel_title)/fig.dpi}
                for fmt in formats:
                    if fmt not in {'png','pdf'}:continue
                    filename=f'{name}.{size}.{fmt}';fig.savefig(output/filename,dpi=dpi);record['files'].append(filename)
                plt.close(fig)
            record.setdefault('sizes',{})[size]={'width_inches':width,'height_inches':height,'dpi':dpi,
                'caption_area_inches':caption_area_inches,'legend_band_inches':decoration_inches,'header_area_inches':top_inches}
        write_json(output/f'{name}.json',record);records.append(record)
    write_json(output/'recipes.json',records)
    return {"directory":str(output.name),"records":[{"name":r['name'],"status":r['status'],"files":r.get('files',[]),"reason":r.get('reason')} for r in records]}

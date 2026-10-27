#!/usr/bin/env python3
"""Render a static PNG overview of a GeoJSON output (for READMEs and papers).

Features far apart (default > 2,000 km) are split into separate panels so
that Europe, North America and South Asia are each legible. Pass a country
polygon layer with ``--basemap`` (e.g. Natural Earth ``admin_0`` shapefile) to
draw borders; without it only a graticule is drawn.

Usage:
    python examples/plot_static_map.py out/demo.geojson docs/sample_map.png \
        --basemap path/to/ne_110m_admin_0_countries.shp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoparser.export import LANGUAGE_COLORS  # noqa: E402
from geoparser.geo_utils import haversine_km  # noqa: E402

MPL_COLORS = {"blue": "#2b6cb0", "red": "#c53030", "green": "#2f855a", "orange": "#dd6b20", "gray": "#4a5568"}


def cluster(features: list[dict[str, Any]], max_km: float) -> list[list[dict[str, Any]]]:
    """Greedy single-link clustering of point features."""
    groups: list[list[dict[str, Any]]] = []
    for f in features:
        lon, lat = f["geometry"]["coordinates"]
        for g in groups:
            if any(
                haversine_km(lat, lon, h["geometry"]["coordinates"][1], h["geometry"]["coordinates"][0]) < max_km
                for h in g
            ):
                g.append(f)
                break
        else:
            groups.append([f])
    return sorted(groups, key=lambda g: g[0]["geometry"]["coordinates"][0])


def main() -> None:
    """Parse arguments and write the PNG."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("geojson", type=Path)
    ap.add_argument("png", type=Path)
    ap.add_argument("--basemap", type=Path, help="Country polygons (any GeoPandas-readable file)")
    ap.add_argument("--cluster-km", type=float, default=2000.0)
    args = ap.parse_args()

    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fc = json.loads(args.geojson.read_text(encoding="utf-8"))
    groups = cluster(fc["features"], args.cluster_km)
    world = gpd.read_file(args.basemap) if args.basemap else None

    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 5.2), squeeze=False)
    for ax, group in zip(axes[0], groups, strict=True):
        lons = [f["geometry"]["coordinates"][0] for f in group]
        lats = [f["geometry"]["coordinates"][1] for f in group]
        pad = max(1.5, 0.25 * max(max(lons) - min(lons), max(lats) - min(lats)))
        x0, x1, y0, y1 = min(lons) - pad, max(lons) + pad, min(lats) - pad, max(lats) + pad
        if world is not None:
            world.plot(ax=ax, color="#f4f1ea", edgecolor="#b8b2a7", linewidth=0.6)
        ax.set_facecolor("#dbe9f4" if world is not None else "white")
        labelled: set[str] = set()
        for f in group:
            p = f["properties"]
            # Default Matplotlib fonts lack Tamil glyphs, so label with the resolved name.
            label = p["resolved_name"].split(",")[0]
            if p["place_type"] == "river":  # a point is a poor proxy for a linear feature
                continue
            color = MPL_COLORS[LANGUAGE_COLORS.get(p["language"], "gray")]
            lon, lat = f["geometry"]["coordinates"]
            if "bbox" in f:
                w, s, e, n = f["bbox"]
                if (e - w) < 3 and (n - s) < 3:  # skip state/country extents
                    ax.add_patch(mpatches.Rectangle((w, s), e - w, n - s, fill=True, alpha=0.12, color=color, lw=0))
            ax.plot(lon, lat, "o", ms=6, color=color, mec="white", mew=0.8, zorder=3)
            if label not in labelled:
                labelled.add(label)
                ax.annotate(label, (lon, lat), xytext=(4, 3), textcoords="offset points", fontsize=8, zorder=4)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.tick_params(labelsize=7)

    handles = [mpatches.Patch(color=MPL_COLORS[c], label=lang) for lang, c in LANGUAGE_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, title="Document language")
    fig.suptitle("Resolved toponyms (point = selected referent, shaded = OSM extent)", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png, dpi=150)
    print(f"Wrote {args.png}")


if __name__ == "__main__":
    main()

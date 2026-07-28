"""Built-in flow graphs reproducing the standard + starless pipelines."""
from __future__ import annotations
from .graph import Graph, Node, Edge, Endpoint


def _e(i, s, sp, d, dp):
    return Edge(f"e{i}", Endpoint(s, sp), Endpoint(d, dp))


_HEAD_NODES = (
    Node("src", "load", {"path": "{input}", "space": "linear-adu"}),
    Node("crop", "crop", {"margin": 40}),
    Node("bg", "background_extract", {"degree": 3, "sample": 12, "pedestal": 0.10}),
    Node("col", "color_calibrate", {"ref_bp_rp": 0.82, "min_stars": 30,
                                    "diagnostic_path": "{out}_pcc_diagnostic.png"}),
    Node("str", "stretch", {"target_bg": 0.18, "shadows_clip": -1.8}),
)
_HEAD_EDGES = (
    _e(1, "src", "image", "crop", "image"),
    _e(2, "crop", "image", "bg", "image"),
    _e(3, "bg", "image", "col", "image"),
    _e(4, "col", "image", "str", "image"),
)


def linear_flow() -> Graph:
    nodes = _HEAD_NODES + (
        Node("fin", "finish", {"saturation": 1.20, "luma_denoise": 0.012,
                               "chroma_denoise": 4.0, "scnr": True}),
        Node("exp", "export_image", {"out_base": "{out}"}),
    )
    edges = _HEAD_EDGES + (
        _e(5, "str", "image", "fin", "image"),
        _e(6, "fin", "image", "exp", "image"),
    )
    return Graph(nodes, edges, name="linear")


def starless_flow() -> Graph:
    nodes = _HEAD_NODES + (
        Node("rs", "remove_stars", {"stride": 256}),
        Node("dn", "masked_denoise", {"bg_luma": 0.06, "bg_chroma": 10.0,
                                      "gal_luma": 0.010, "gal_chroma": 3.0, "feather": 25.0}),
        Node("sat", "saturate", {"saturation": 1.20}),
        Node("rec", "screen_recombine", {}),
        Node("exp", "export_image", {"out_base": "{out}"}),
        Node("pv_sl", "preview_sink", {"out_path": "{out}_starless.png", "stretch": False}),
        Node("pv_st", "preview_sink", {"out_path": "{out}_starlayer.png", "stretch": False}),
    )
    edges = _HEAD_EDGES + (
        _e(5, "str", "image", "rs", "image"),
        _e(6, "rs", "starless", "dn", "image"),
        _e(7, "dn", "image", "sat", "image"),
        _e(8, "sat", "image", "rec", "base"),
        _e(9, "rs", "stars", "rec", "overlay"),
        _e(10, "rec", "image", "exp", "image"),
        _e(11, "sat", "image", "pv_sl", "image"),
        _e(12, "rs", "stars", "pv_st", "image"),
    )
    return Graph(nodes, edges, name="starless")

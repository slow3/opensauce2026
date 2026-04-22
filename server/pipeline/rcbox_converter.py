"""
.rcbox → LichtFeld bounding region converter.

Parses RealityCapture's .rcbox XML format and extracts the
axis-aligned bounding box in world coordinates.

NOTE: The `s` (scale) parameter in the Residual element is RC's
internal SfM scale factor. Its exact relationship to world-space
units has NOT been definitively confirmed from official docs.
The widthHeightDepth values appear to be in mm based on practical
measurements (a 2438 x 2504 x 1589 volume matches a ~2.4m human scan space).

ACTION REQUIRED (Senior dev): Verify scale by measuring a known
distance in a calibrated scan and comparing to the rcbox dimensions
before using this converter in production.
"""

import xml.etree.ElementTree as ET
import numpy as np
import logging
import struct

log = logging.getLogger("rcbox")


def parse_rcbox(rcbox_path: str) -> dict:
    """
    Parse a .rcbox file and return the bounding box parameters.

    Returns:
        {
            "center":    [x, y, z],       # centre in RC coordinate space
            "size":      [w, h, d],       # widthHeightDepth (likely mm)
            "rotation":  [[r00,...], ...], # 3x3 rotation matrix
            "scale":     float,           # RC internal scale factor s
            "translate": [tx, ty, tz],    # translation vector
        }
    """
    tree = ET.parse(rcbox_path)
    root = tree.getroot()

    # widthHeightDepth
    whd_str = root.get("widthHeightDepth", "1 1 1")
    w, h, d = [float(v) for v in whd_str.split()]

    # Centre
    centre_el = root.find("CentreEuclid")
    if centre_el is not None:
        centre_str = centre_el.get("centre") or centre_el.text or "0 0 0"
    else:
        centre_str = "0 0 0"
    cx, cy, cz = [float(v) for v in centre_str.strip().split()]

    # Residual
    residual = root.find("Residual")
    scale = 1.0
    R     = np.eye(3)
    t     = np.zeros(3)

    if residual is not None:
        scale = float(residual.get("s", 1.0))
        r_str = residual.find("R").text if residual.find("R") is not None else None
        t_str = residual.find("t").text if residual.find("t") is not None else None

        if r_str:
            vals = [float(v) for v in r_str.strip().split()]
            R = np.array(vals).reshape(3, 3)
        if t_str:
            t = np.array([float(v) for v in t_str.strip().split()])

    return {
        "center":    [cx, cy, cz],
        "size":      [w, h, d],
        "rotation":  R.tolist(),
        "scale":     scale,
        "translate": t.tolist(),
    }


def rcbox_to_lichtfeld_region(rcbox_path: str) -> dict:
    """
    Convert .rcbox to a LichtFeld-compatible bounding box dict.

    LichtFeld expects a region as: center (x,y,z) + half-extents (x,y,z).
    We treat widthHeightDepth as the full extents of the box.

    TODO: Update this once LichtFeld's exact region format is confirmed
    from their Python plugin API docs.
    """
    box = parse_rcbox(rcbox_path)
    w, h, d = box["size"]

    return {
        "center": box["center"],
        "half_extents": [w / 2.0, h / 2.0, d / 2.0],
        "rotation": box["rotation"],
        "_note": "Units unverified — confirm s-factor with senior dev before production use",
        "_rcbox_scale": box["scale"],
    }


def trim_ply_with_rcbox(input_ply: str, rcbox_path: str, output_ply: str):
    """
    Trim a Gaussian Splat .ply to the reconstruction region defined in the .rcbox.

    Approach: parse the bounding box, then filter splat points
    that fall outside it. Uses numpy for speed.

    Requires: numpy, plyfile (pip install plyfile)
    """
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        raise ImportError("plyfile required: pip install plyfile")

    log.info(f"Trimming {input_ply} with {rcbox_path}")

    box   = parse_rcbox(rcbox_path)
    cx, cy, cz = box["center"]
    w, h, d    = box["size"]
    R          = np.array(box["rotation"])

    ply  = PlyData.read(input_ply)
    verts = ply["vertex"]

    x = np.array(verts["x"])
    y = np.array(verts["y"])
    z = np.array(verts["z"])

    pts = np.stack([x - cx, y - cy, z - cz], axis=1)  # translate to box centre
    local = pts @ R  # rotate into box-aligned space

    mask = (
        (np.abs(local[:, 0]) <= w / 2.0) &
        (np.abs(local[:, 1]) <= h / 2.0) &
        (np.abs(local[:, 2]) <= d / 2.0)
    )

    log.info(f"Splat trim: {mask.sum()} / {len(mask)} gaussians kept")

    # Rebuild filtered vertex data
    filtered = {prop.name: np.array(verts[prop.name])[mask]
                for prop in verts.properties}

    el = PlyElement.describe(
        np.array(list(zip(*filtered.values())),
                 dtype=[(k, v.dtype) for k, v in filtered.items()]),
        "vertex"
    )
    PlyData([el], text=False).write(output_ply)
    log.info(f"Trimmed splat written: {output_ply}")

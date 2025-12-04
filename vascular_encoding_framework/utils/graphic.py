"""Utility plotting helpers used by the CLI wrappers.

The legacy CLI expects a `plot_adapted_frame` function inside
`vascular_encoding_framework.utils`, so we provide a thin wrapper that
delegates to the Centerline/CenterlineTree `plot_adapted_frame` method.
"""

from __future__ import annotations

from typing import Any

import pyvista as pv


def plot_adapted_frame(cntrln, vmesh=None, scale: float = 1.0, show: bool = True, **kwargs: Any):
    """
    Delegate to the `plot_adapted_frame` method on the provided centerline object.

    Parameters
    ----------
    cntrln : Centerline or CenterlineTree
        Object exposing a `plot_adapted_frame` method.
    vmesh : VascularMesh or pv.PolyData, optional
        Optional vascular mesh to render alongside the centerline.
    scale : float, optional
        Scale factor for glyphs.
    show : bool, optional
        Whether to render the plot immediately.
    **kwargs : Any
        Forwarded to the underlying method.
    """

    if not hasattr(cntrln, "plot_adapted_frame"):
        raise AttributeError("Provided centerline object has no 'plot_adapted_frame' method.")

    return cntrln.plot_adapted_frame(vmesh=vmesh, scale=scale, show=show, **kwargs)

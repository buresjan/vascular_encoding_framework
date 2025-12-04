"""Lightweight plotting helpers used by the CLI wrappers."""

from __future__ import annotations

import pyvista as pv


def plot_adapted_frame(cntrln, vmesh=None, **kwargs):
    """
    Proxy to ``CenterlineTree.plot_adapted_frame`` for CLI helpers.

    Parameters
    ----------
    cntrln : Centerline or CenterlineTree
        The centerline object to visualize.
    vmesh : pv.PolyData, optional
        Optional vascular mesh to provide context in the plot.
    **kwargs : dict
        Extra keyword arguments forwarded to ``plot_adapted_frame``.
    """

    if hasattr(cntrln, "plot_adapted_frame"):
        return cntrln.plot_adapted_frame(vmesh=vmesh, **kwargs)

    raise AttributeError("Provided object has no 'plot_adapted_frame' method.")

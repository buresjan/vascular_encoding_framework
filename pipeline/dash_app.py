from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import dash
from dash import Dash, Input, Output, State, callback, dcc, html, no_update
import plotly.graph_objects as go
import pyvista as pv

from pipeline import run_pipeline
from pipeline.vef_pipeline import DEFAULT_RUN_KWARGS


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_mesh(mesh_path: Path) -> pv.PolyData:
    mesh = pv.read(str(mesh_path))
    return mesh.triangulate()


def mesh_to_figure(mesh: pv.PolyData) -> go.Figure:
    if mesh.n_faces == 0:
        return empty_figure("Mesh is empty.")
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    pts = mesh.points
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#4cc9f0",
                opacity=0.9,
                flatshading=True,
                lighting=dict(ambient=0.45, diffuse=0.65, specular=0.45, roughness=0.7),
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="#9fb3c8"),
            )
        ],
    )
    return fig


NUMBER_FIELDS = [
    dict(key="tau0", label="tau0 (axial)", step=0.05),
    dict(key="theta0", label="theta0 (angle)", step=0.05),
    dict(key="bump_amp", label="bump_amp", step=0.05),
    dict(key="sigma_t", label="sigma_t", step=0.01),
    dict(key="sigma_theta", label="sigma_theta", step=0.01),
    dict(key="rim_tol", label="rim_tol", step=1e-4),
    dict(key="deform_r1", label="deform_r1", step=0.5),
    dict(key="deform_r2", label="deform_r2", step=0.5),
    dict(key="clip_offset", label="clip_offset", step=0.1),
    dict(key="hole_size", label="hole_size", step=10),
    dict(key="voxel_pitch", label="voxel_pitch", step=0.1),
]

INITIALS: Dict[str, Any] = {k: DEFAULT_RUN_KWARGS[k] for k in DEFAULT_RUN_KWARGS}
INITIAL_OUTPUT_DIR = str(Path(INITIALS["output_dir"]))

app = Dash(
    __name__,
    title="VEF Pipeline Dash",
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&display=swap"
    ],
    suppress_callback_exceptions=True,
)
server = app.server


def number_control(key: str, label: str, step: float) -> html.Div:
    return html.Div(
        className="control",
        children=[
            html.Label(label, htmlFor=key),
            dcc.Input(
                id=key,
                type="number",
                value=INITIALS[key],
                step=step,
                debounce=True,
                className="control-input",
            ),
        ],
    )


app.layout = html.Div(
    className="page",
    children=[
        html.Link(
            rel="preconnect",
            href="https://fonts.googleapis.com",
        ),
        html.Link(
            rel="preconnect",
            href="https://fonts.gstatic.com",
            crossOrigin="anonymous",
        ),
        html.Div(
            className="hero",
            children=[
                html.Div(
                    children=[
                        html.Div("VEF pipeline playground", className="title"),
                        html.Div(
                            "Tweak the bump/deform parameters from vef_pipeline.py and render the STL instantly.",
                            className="subtitle",
                        ),
                    ]
                ),
                html.Div("Outputs land in 'outputs/'", className="pill"),
            ],
        ),
        html.Div(
            className="grid",
            children=[
                html.Div(
                    className="panel",
                    children=[
                        html.Div("Inputs", className="subtitle", style={"color": "#c2d5ee"}),
                        html.Div(
                            className="controls-grid",
                            children=[
                                number_control(f["key"], f["label"], f["step"])
                                for f in NUMBER_FIELDS
                            ],
                        ),
                        html.Div(
                            className="control",
                            style={"marginTop": "12px"},
                            children=[
                                html.Label("output_dir"),
                                dcc.Input(
                                    id="output_dir",
                                    type="text",
                                    value=INITIAL_OUTPUT_DIR,
                                    debounce=True,
                                    className="control-input",
                                ),
                                html.Small("Relative paths are resolved from the repo root."),
                            ],
                        ),
                        html.Div(
                            className="actions",
                            children=[
                                html.Button(
                                    "Run pipeline",
                                    id="run-btn",
                                    n_clicks=0,
                                    className="button primary",
                                ),
                                html.Button(
                                    "Download STL",
                                    id="download-btn",
                                    n_clicks=0,
                                    className="button secondary",
                                    disabled=True,
                                ),
                            ],
                        ),
                        html.Div(id="run-status", className="status"),
                    ],
                ),
                html.Div(
                    className="panel",
                    children=[
                        html.Div("Preview", className="subtitle", style={"color": "#c2d5ee"}),
                        html.Div(
                            className="graph-wrapper",
                            children=[
                                dcc.Graph(
                                    id="mesh-graph",
                                    figure=empty_figure("Adjust parameters and run the pipeline."),
                                    config={"displaylogo": False, "scrollZoom": True},
                                    style={"height": "520px"},
                                )
                            ],
                        ),
                    ],
                ),
            ],
        ),
        dcc.Store(id="last-result"),
        dcc.Download(id="download-stl"),
    ],
)


@callback(
    Output("mesh-graph", "figure"),
    Output("run-status", "children"),
    Output("last-result", "data"),
    Output("download-btn", "disabled"),
    Input("run-btn", "n_clicks"),
    State("tau0", "value"),
    State("theta0", "value"),
    State("bump_amp", "value"),
    State("sigma_t", "value"),
    State("sigma_theta", "value"),
    State("rim_tol", "value"),
    State("deform_r1", "value"),
    State("deform_r2", "value"),
    State("clip_offset", "value"),
    State("hole_size", "value"),
    State("voxel_pitch", "value"),
    State("output_dir", "value"),
    prevent_initial_call=True,
)
def run_pipeline_from_inputs(
    _n_clicks,
    tau0,
    theta0,
    bump_amp,
    sigma_t,
    sigma_theta,
    rim_tol,
    deform_r1,
    deform_r2,
    clip_offset,
    hole_size,
    voxel_pitch,
    output_dir,
):
    try:
        out_dir = Path(str(output_dir or INITIAL_OUTPUT_DIR)).expanduser().resolve()
    except Exception:
        out_dir = Path(INITIAL_OUTPUT_DIR).resolve()

    params = {
        "tau0": _float(tau0, INITIALS["tau0"]),
        "theta0": _float(theta0, INITIALS["theta0"]),
        "bump_amp": _float(bump_amp, INITIALS["bump_amp"]),
        "sigma_t": _float(sigma_t, INITIALS["sigma_t"]),
        "sigma_theta": _float(sigma_theta, INITIALS["sigma_theta"]),
        "rim_tol": _float(rim_tol, INITIALS["rim_tol"]),
        "deform_r1": _float(deform_r1, INITIALS["deform_r1"]),
        "deform_r2": _float(deform_r2, INITIALS["deform_r2"]),
        "clip_offset": _float(clip_offset, INITIALS["clip_offset"]),
        "hole_size": _float(hole_size, INITIALS["hole_size"]),
        "voxel_pitch": _float(voxel_pitch, INITIALS["voxel_pitch"]),
        "output_dir": out_dir,
        "encoding_path": INITIALS["encoding_path"],
        "vcs_map_path": INITIALS["vcs_map_path"],
        "partner_stl": INITIALS["partner_stl"],
        "partner_orig_rim": INITIALS["partner_orig_rim"],
    }

    try:
        final_path, uid = run_pipeline(**params)
        mesh = load_mesh(final_path)
        fig = mesh_to_figure(mesh)
        status = [
            html.Div([html.Strong("Saved: "), str(final_path)]),
            html.Div([html.Strong("UID: "), uid]),
            html.Div(
                [
                    html.Strong("Mesh: "),
                    f"{mesh.n_faces:,} faces · {mesh.n_points:,} points",
                ]
            ),
        ]
        data = {"path": str(final_path), "uid": uid}
        return fig, status, data, False
    except Exception as exc:
        status = [
            html.Div(html.Strong("Pipeline failed")),
            html.Div(str(exc)),
        ]
        return (
            empty_figure("Pipeline failed. Check the status for details."),
            status,
            no_update,
            True,
        )


@callback(
    Output("download-stl", "data"),
    Input("download-btn", "n_clicks"),
    State("last-result", "data"),
    prevent_initial_call=True,
)
def download_last_result(_n_clicks, data):
    if not data or "path" not in data:
        return no_update
    mesh_path = Path(data["path"])
    if not mesh_path.exists():
        return no_update
    return dcc.send_file(mesh_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)

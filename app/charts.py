import numpy as np
import plotly.graph_objects as go

from engine.schemas import OptimizationResult, Scenario


def room_figure(scenario: Scenario, result: OptimizationResult, three_dimensional=False):
    positions = [
        (
            rack.x_m if rack.x_m is not None else 1 + 2 * index,
            rack.y_m if rack.y_m is not None else 1,
        )
        for index, rack in enumerate(scenario.rack_clusters)
    ]
    temperatures = [rack.inlet_temp_c for rack in result.optimized.racks]
    names = [rack.cluster_id for rack in result.optimized.racks]
    x, y = np.asarray(positions).T
    if not three_dimensional:
        xx, yy = np.meshgrid(
            np.linspace(max(0, min(x) - 1), max(x) + 1, 40),
            np.linspace(max(0, min(y) - 1), max(y) + 1, 30),
        )
        weights = 1 / np.maximum((xx[..., None] - x) ** 2 + (yy[..., None] - y) ** 2, 0.01)
        field = np.sum(weights * temperatures, axis=-1) / np.sum(weights, axis=-1)
        figure = go.Figure(
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=field,
                colorscale="Cividis",
                colorbar_title="Inlet °C",
                contours={"coloring": "heatmap", "showlines": False},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                text=names,
                marker={"size": 10, "color": "#172d27", "line": {"color": "white", "width": 2}},
                name="Rack centers",
            )
        )
        for rx, ry, name in zip(x, y, names):
            figure.add_annotation(
                x=rx,
                y=ry,
                text=name,
                yshift=18,
                showarrow=False,
                bgcolor="white",
                borderpad=3,
                font={"color": "#172d27"},
            )
        figure.update_layout(xaxis_title="Room x (m)", yaxis_title="Room y (m)")
    else:
        figure = go.Figure()
        for index, ((rx, ry), temperature, name) in enumerate(zip(positions, temperatures, names)):
            # The example envelope is 0.6 x 1.2 x 2 m; color encodes rack inlet, not a CFD volume.
            figure.add_trace(
                go.Mesh3d(
                    x=[rx - 0.3, rx + 0.3, rx + 0.3, rx - 0.3] * 2,
                    y=[ry - 0.6, ry - 0.6, ry + 0.6, ry + 0.6] * 2,
                    z=[0] * 4 + [2] * 4,
                    i=[0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3],
                    j=[1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4],
                    k=[2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7],
                    intensity=[temperature] * 8,
                    colorscale="Cividis",
                    cmin=18,
                    cmax=35,
                    showscale=index == 0,
                    colorbar_title="Inlet °C",
                    name=name,
                    hovertemplate=f"{name}<br>Inlet {temperature:.1f} °C<extra></extra>",
                )
            )
        figure.update_layout(
            scene={
                "xaxis_title": "Room x (m)",
                "yaxis_title": "Room y (m)",
                "zaxis_title": "Height (m)",
                "aspectmode": "data",
            }
        )
    figure.update_layout(
        height=420, font={"color": "#172d27"}, margin={"l": 0, "r": 0, "t": 15, "b": 0}
    )
    return figure


def hotspot_rows(result: OptimizationResult, timestep_seconds: float):
    rows = []
    for step in result.plan:
        for rack in step.plant.racks:
            if rack.inlet_temp_c >= 26 or rack.delta_t_c >= 14:
                rows.append(
                    {
                        "Minute": (step.step + 1) * timestep_seconds / 60,
                        "Cluster": rack.cluster_id,
                        "Inlet (°C)": rack.inlet_temp_c,
                        "Rack rise (K)": rack.delta_t_c,
                        "Inlet headroom (K)": 27 - rack.inlet_temp_c,
                    }
                )
    return rows


def time_series_figure(frame, y_label: str):
    figure = go.Figure()
    for column in frame.columns:
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                name=column,
                mode="lines+markers",
                marker={"size": 6},
            )
        )
    figure.update_layout(
        height=240,
        font={"color": "#172d27"},
        xaxis_title=frame.index.name,
        yaxis_title=y_label,
        margin={"l": 0, "r": 0, "t": 15, "b": 0},
    )
    return figure

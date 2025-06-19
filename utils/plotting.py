"""Plotting Utils."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Color Mapping
COLOR_MAP = {
    "Car": "#1f77b4",             # Blue
    "Van": "#17becf",             # Cyan
    "Truck": "#2ca02c",           # Green
    "Pedestrian": "#bcbd22",      # Olive
    "Person_sitting": "#8c564b",  # Brown
    "Cyclist": "#9467bd",         # Purple
    "Tram": "#7f7f7f",            # Gray
    "Misc": "#e377c2",            # Pink
    "DontCare": "#d62728",         # (Muted Red – optional, or change)

    "Car_P": "#00bfff",            # Bright Sky Blue
    "Van_P": "#00ffff",            # Aqua
    "Truck_P": "#00ff00",          # Lime Green
    "Pedestrian_P": "#ffff00",     # Yellow
    "Person_sitting_P": "#ff8c00", # Dark Orange
    "Cyclist_P": "#da70d6",        # Orchid
    "Tram_P": "#c0c0c0",           # Silver
    "Misc_P": "#ff69b4",           # Hot Pink
    "DontCare_P": "#ff4500"        # Orange Red
}


def get_figure(rows: int = 1, cols: int = 1, titles: list[str] = None) -> go.Figure:
    """Return a Plotly Figure.

    Returns:
        go.Figure: Plotly Figure
    """
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{'type': 'scene'}] * cols] * rows,
        subplot_titles=titles,
    )
    for row in range(1, rows + 1):
        for col in range(1, cols+1):
            fig.update_scenes(
                xaxis=dict(showgrid=False, zeroline=False, showbackground=True, backgroundcolor='black'),
                yaxis=dict(showgrid=False, zeroline=False, showbackground=True, backgroundcolor='black'),
                zaxis=dict(showgrid=False, zeroline=False, showbackground=True, backgroundcolor='black'),
                aspectmode='data',
                row=row,
                col=col,
            )
    fig.update_layout(
        paper_bgcolor ="black",
    )
    return fig


def plot_point_cloud(data: np.ndarray, name: str, fig: go.Figure, row: int = None, col: int = None) -> None:
    """Plot 3D Point Cloud.

    Args:
        data (np.ndarray): Point Cloud data
        fig (go.Figure): Plotly Figure
    """
    fig.add_trace(
        go.Scatter3d(
            x=data[:, 0], 
            y=data[:, 1],
            z=data[:, 2],
            mode="markers",
            marker=dict(
                size=1,
                color=data[:, 2],
                colorscale="reds",
            ),
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    # add text
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="text",
            text=[name],
            textfont=dict(
                family="Arial Black",  # Font family
                size=14,               # Font size
                color="#ff0000",            # Font color
            ),
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def plot_3d_bbox(points_3d: np.ndarray, name: str, fig: go.Figure, extra_text: str = "", row: int = None, col: int = None) -> None:
    """Plot 3D Bounding Box.

    Args:
        points_3d (np.ndarray): 3D Points
        name (str): Name of the object
        fig (go.Figure): Plotly Figure
    """
    # Define edges of the box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    bbox = []
    for s, e in edges:
        x, y, z = zip(points_3d[s], points_3d[e])
        bbox.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(
                    color=COLOR_MAP[name],
                    width=3,
                ),
                hovertext=[],
                hoverinfo="text",
                hoverlabel=dict(
                    font=dict(size=20, color=COLOR_MAP[name]),
                    bgcolor='black'
                ),
                showlegend=False,
            )
        )
    fig.add_traces(bbox, rows=row, cols=col)
    # add text
    x, y, z = points_3d[4, :3]
    fig.add_trace(
        go.Scatter3d(
            x=[x],
            y=[y],
            z=[z*1.05],
            mode="text",
            text=[f"{name}_{extra_text}" if len(extra_text) else name],
            textfont=dict(
                family="Arial Black",  # Font family
                size=14,               # Font size
                color="#ffffff",            # Font color
            ),
            showlegend=False,
        ),
        row=row,
        col=col,
    )
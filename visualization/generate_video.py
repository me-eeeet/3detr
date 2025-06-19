"""Generate Point Cloud Visualization Video."""

import sys
sys.path.append("../")


from pathlib import Path
from argparse import ArgumentParser
from tqdm.auto import tqdm

import numpy as np
import plotly.graph_objects as go
from moviepy import ImageSequenceClip

from datasets import KITTIDatasetConfig


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


def read_point_cloud(path: Path) -> np.ndarray:
    """Read Point Cloud File.

    Args:
        path (Path): Path to the point cloud file

    Returns:
        np.ndarray: Point Cloud Data
    """
    return np.fromfile(str(path), dtype=np.float32).reshape((-1, 4))


def preprocess_point_cloud(point_cloud: np.ndarray) -> np.ndarray:
    max_bounds = np.max(np.abs(point_cloud), axis=0)
    pad_points = np.zeros((4, point_cloud.shape[1]), dtype=point_cloud.dtype)
    k = 0
    for i in [-1, 1]:
        for j in [-1, 1]:
            pad_points[k, :3] = [i * max_bounds[0], j * max_bounds[1], i * 100]
            k += 1
    point_cloud = np.r_[point_cloud, pad_points]
    # FOV Filtering
    return point_cloud


def create_animation(frames: list[go.Frame]) -> go.Figure:

    # Define camera parameters
    camera = dict(
        eye=dict(x=0, y=0, z=0.2),    # Camera position
        center=dict(x=0, y=0, z=0),       # Look-at point
        up=dict(x=0, y=0, z=1),            # Up vector
    )

    fig = go.Figure(
        # data=frames[0].data,
        layout=go.Layout(
            scene_camera=camera,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showbackground=True, backgroundcolor='black'),
                yaxis=dict(showgrid=False, zeroline=False, showbackground=True, backgroundcolor='black'),
                zaxis=dict(showgrid=False, zeroline=False, showbackground=True, backgroundcolor='black'),
                aspectmode='data',
            ),
            paper_bgcolor ="black",
            # updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])]
        ),
        # frames=frames,
    )
    return fig


def plot_predictions(predictions: dict[str, np.ndarray], dataset_config: KITTIDatasetConfig) -> list:
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    bboxes = []
    n = len(predictions["ids"])
    for idx in range(n):
        name = dataset_config.class2type[predictions["ids"][idx]]
        points_3d = predictions["corners"][idx]
        score = predictions["scores"][idx]
        if name in {"Car"} and score < 0.8:
            continue
        elif score < 0.5:
            continue
        # Define edges of the box
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
        bboxes.extend(bbox)
        # Add Name of the Class
        x, y, z = points_3d[4, :3]
        bboxes.append(
            go.Scatter3d(
                x=[x],
                y=[y],
                z=[z*1.05],
                mode="text",
                text=[name],
                textfont=dict(
                    family="Arial Black",  # Font family
                    size=14,               # Font size
                    color="#ffffff",     # Font color
                ),
                showlegend=False,
            )
        )
    return bboxes


def save_video(
    fig: go.Figure,
    frames: list[go.Frame],
    output_path: Path,
    fps: int = 10,
) -> None:
    
    temp_dir = Path("temp_frames")
    temp_dir.mkdir(parents=True, exist_ok=True)

    frames_paths = []
    for i, frame in tqdm(enumerate(frames), desc="Saving Frames"):
        fig.update(data=frame.data)
        path = str(temp_dir / f"frame_{i}.png")
        fig.write_image(path, width=1920, height=1080)
        frames_paths.append(path)
    
    # Create & Save the Video
    print("Writing Video...!!!")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip = ImageSequenceClip(sequence=frames_paths, fps=fps)
    clip.write_videofile(output_path)

    # Save as a gif
    print("Writing GIF...!!!")
    clip.write_gif(output_path.with_suffix(".gif"))


def generate_video(
    root_dir: Path,
    predictions_dir: Path,
    output_path: Path,
    fps: int = 10,
    show_animation: bool = False,
) -> None:
    
    point_cloud_dir = root_dir / "velodyne_points/data"

    # Dataset Config
    dataset_config = KITTIDatasetConfig()

    # Read & process point clouds
    frames = []
    for path in tqdm(sorted(point_cloud_dir.glob("*.bin")), desc="Preparing Frame"):
        
        name = path.stem
        point_cloud = read_point_cloud(path=path)

        # Preprocess the Point cloud
        point_cloud = preprocess_point_cloud(point_cloud=point_cloud)
        
        # Plot Point Cloud
        point_could_plot = go.Scatter3d(
            x=point_cloud[:, 0], 
            y=point_cloud[:, 1],
            z=point_cloud[:, 2],
            mode="markers",
            marker=dict(
                size=1,
                color=point_cloud[:, 3],
                colorscale="reds",
            ),
            showlegend=False,
        )

        # Read Predictions
        predictions: dict[str, np.ndarray] = np.load(predictions_dir / f"{name}.npy", allow_pickle=True).item()
        bboxes = plot_predictions(predictions=predictions, dataset_config=dataset_config)
        
        frame = go.Frame(data=[point_could_plot] + bboxes)
        frames.append(frame)
    
    # Create Animation Figure
    fig = create_animation(frames=frames)
    if show_animation:
        fig.show()

    # Save Video
    save_video(fig=fig, frames=frames, output_path=output_path, fps=fps)


if __name__ == "__main__":

    parser = ArgumentParser(description="KITTI Video Generator")
    parser.add_argument(
        "-r",
        "--root_dir",
        type=Path,
        help="KITTI continuous raw data path",
        required=True,
    )

    parser.add_argument(
        "-p",
        "--predictions_dir",
        type=Path,
        help="Predictions Path",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        help="Ouput Video Path",
        required=True,
    )

    parser.add_argument(
        "-f",
        "--fps",
        type=int,
        help="Ouput Video FPS",
        default=10,
    )

    parser.add_argument(
        "-a",
        "--show_animation",
        action="store_true",
        help="Display Animation",
        default=False
    )

    args = parser.parse_args()
    generate_video(
        root_dir=args.root_dir,
        output_path=args.output_path,
        predictions_dir=args.predictions_dir,
        fps=args.fps,
        show_animation=args.show_animation,
    )
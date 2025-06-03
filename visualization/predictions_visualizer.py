"""Streamlit Application to View Predictions."""

import sys
sys.path.append("../")

import streamlit as st

from datasets.kitti import KITTIDatasetConfig, KITTIDetectionDataset
from visualization.plot_predictions import get_plot


def prepare_dataset() -> None:
    st.session_state.dataset_config  = KITTIDatasetConfig()
    st.session_state.dataset = KITTIDetectionDataset(
        dataset_config=st.session_state.dataset_config,
        split_set=st.session_state.split_set,
        root_dir=st.session_state.root_dir,
        radius=float(st.session_state.radius),
        augment=False,
    )

def go_previous() -> None:
    st.session_state.curr_index = int((st.session_state.curr_index - 1) % len(st.session_state.dataset))

def go_next() -> None:
    st.session_state.curr_index = int((st.session_state.curr_index + 1) % len(st.session_state.dataset))

def update_curr_index() -> None:
    st.session_state.curr_index = min(int(st.session_state.index), len(st.session_state.dataset))


st.set_page_config(layout="wide")
st.title("Predictions Visualization")

st.sidebar.text_input("Dataset Directory", "/common/dataset/kitti/object", key="root_dir", on_change=prepare_dataset)
st.sidebar.selectbox("Split", ["val", "train", "trainval"], key="split_set", on_change=prepare_dataset)
st.sidebar.text_input("Radius", 15, key="radius", on_change=prepare_dataset)
st.sidebar.text_input("Predictions Directory", "../predictions/rc_gt_aug_3_class", key="predictions_dir")
st.sidebar.text_input("# Points", 3072, key="n_points")
st.sidebar.text_input("Index", 0, key="index", on_change=update_curr_index)

if "curr_index" not in st.session_state:
    st.session_state.curr_index = 0

if "dataset" not in st.session_state:
    prepare_dataset()

col1, col2 = st.columns(2)
with col1:
    st.button("⬅️ Previous", on_click=go_previous)

with col2:
    st.button("Next ➡️", on_click=go_next)

if st.session_state.predictions_dir:
    fig = get_plot(
        index=st.session_state.curr_index,
        predictions_dir=st.session_state.predictions_dir,
        n_points=int(st.session_state.n_points),
        dataset_config=st.session_state.dataset_config,
        dataset=st.session_state.dataset,
    )
    st.plotly_chart(fig)
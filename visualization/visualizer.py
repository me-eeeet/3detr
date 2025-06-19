"""Streamlit Application to View Predictions."""

import sys
sys.path.append("../")

import streamlit as st

from datasets.kitti import KITTIDatasetConfig, KITTIDetectionDataset
from visualization.plot_raw import get_plot


def prepare_dataset() -> None:
    st.session_state.dataset_config  = KITTIDatasetConfig()
    st.session_state.dataset = KITTIDetectionDataset(
        dataset_config=st.session_state.dataset_config,
        split_set=st.session_state.split_set,
        root_dir=st.session_state.root_dir,
        augment=False,
    )

def go_previous() -> None:
    st.session_state.curr_index = int((st.session_state.curr_index - 1) % len(st.session_state.dataset))

def go_next() -> None:
    st.session_state.curr_index = int((st.session_state.curr_index + 1) % len(st.session_state.dataset))

def update_curr_index() -> None:
    st.session_state.curr_index = min(int(st.session_state.index), len(st.session_state.dataset))


st.set_page_config(layout="wide")
st.title("Data Visualization")

st.sidebar.text_input("Dataset Directory", "/common/dataset/kitti/object", key="root_dir", on_change=prepare_dataset)
st.sidebar.selectbox("Split", ["train", "val", "trainval"], key="split_set", on_change=prepare_dataset)
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

fig = get_plot(
    index=st.session_state.curr_index,
    dataset_config=st.session_state.dataset_config,
    dataset=st.session_state.dataset,
)
st.plotly_chart(fig)
import streamlit as st
from table_segmentation import TableSegmentation
import utils
import numpy as np
ts = TableSegmentation()
image = ts.img
gray = ts.gray.copy()
col1, col2 = st.columns(2)

with col1:
    st.header("Original Image")
    st.subheader("Original")
    st.image(image)

with col2:
    st.header("Grayscale Image")
    st.subheader("Copy")
    result = gray
    st.image(ts.gray)

thresh = ts.thresholding()


mask = ts.empty_mask()
st.header("Segmentation result")
with st.container():
    horizontal_lines_col, vertical_lines_col = st.columns(2)
    mask_with_horizontal_lines = mask.copy()
    mask_with_vertical_lines = mask.copy()
    with horizontal_lines_col:
        st.header("horizontal_lines_col")
        horizontal_lines_col.image(ts.horizontal_lines(mask_with_horizontal_lines))

    with vertical_lines_col:
        st.header("vertical_lines_col")
        st.image(ts.vertical_lines(mask_with_vertical_lines))


with st.container():
    st.header("Overlay")
    with st.container():
        horizontal_lines_col, vertical_lines_col = st.columns(2)
        horizontal_lines_col.markdown(utils.make_overlay(image, mask_with_horizontal_lines), unsafe_allow_html=True)
        vertical_lines_col.markdown(utils.make_overlay(image, mask_with_vertical_lines), unsafe_allow_html=True)
st.header("Overlay")
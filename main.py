import streamlit as st
from table_segmentation import TableSegmentation
import utils

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

st.header("Segmentation result")
with st.container() as container:
    horizontal_lines_col, vertical_lines_col = st.columns(2)
    st.write("This is inside the container")
    with horizontal_lines_col:
        st.header("horizontal_lines_col")
        horizontal_lines_col.image(ts.horizontal_lines())

    with vertical_lines_col:
        st.header("vertical_lines_col")
        st.image(ts.vertical_lines())

st.write("""
# Overlay
Threshold *Image!*
""")
st.markdown(utils.make_overlay(image, thresh), unsafe_allow_html=True)

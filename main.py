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

st.write("""
# Table Segmentation
Threshold *Image!*
""")
st.image(thresh)
st.image(ts.horizontal_lines())

st.write("""
# Overlay
Threshold *Image!*
""")
st.markdown(utils.make_overlay(image, thresh), unsafe_allow_html=True)
st.write("""
# Table Segmentation
Threshold *Image!*
""")
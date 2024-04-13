import streamlit as st
from table_segmentation import TableSegmentation

ts = TableSegmentation()
st.write("""
# Table Segmentation
Original *Image!*
""")
image = ts.img
st.image(image)

result = image.copy()


st.write("""
# Table Segmentation
Gray *Image!*
""")
st.image(ts.gray)
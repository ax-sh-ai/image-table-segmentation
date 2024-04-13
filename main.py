import imutils
import streamlit as st
from table_segmentation import TableSegmentation
import utils

ts = TableSegmentation()
image = ts.img
gray = ts.gray.copy()
cropped_table_image = ts.find_and_crop_biggest_region(gray)

# beginning

col1, col2 = st.columns(2)

with col1:
    st.header("Original Image")
    st.text("Original")
    st.image(image)

with col2:
    st.header("Grayscale Image")
    st.text("cropped biggest region")
    st.image(cropped_table_image)


overlay_image = cropped_table_image
overlay_mask = ts.pic.empty_mask(image)

st.header("Segmentation result")

mask_with_horizontal_lines = overlay_mask.copy()
mask_with_vertical_lines = overlay_mask.copy()

with st.container():
    horizontal_lines_col, vertical_lines_col = st.columns(2)
    with horizontal_lines_col:
        st.text("Detected horizontal lines")
        horizontal_lines_col.image(ts.horizontal_lines(mask_with_horizontal_lines))

    with vertical_lines_col:
        st.text("Detected vertical lines")
        st.image(ts.vertical_lines(mask_with_vertical_lines))


cropped_table_mask_with_horizontal_lines = ts.pic.empty_mask(cropped_table_image)
with st.container():
    st.header("Overlay")
    with st.container():
        horizontal_lines_col, vertical_lines_col = st.columns(2)

        horizontal_lines_col.markdown(
            utils.make_overlay(
                cropped_table_image, cropped_table_mask_with_horizontal_lines
            ),
            unsafe_allow_html=True,
        )
        vertical_lines_col.markdown(
            utils.make_overlay(cropped_table_image, mask_with_vertical_lines),
            unsafe_allow_html=True,
        )

st.header("Segmentation")

for i in ts.horizontal_segments():
    st.image(i)

# st.header("edgeMap")
# edgeMap = imutils.auto_canny(gray)
# st.image(edgeMap)

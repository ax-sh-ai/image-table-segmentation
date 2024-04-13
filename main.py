import imutils
import streamlit as st
from table_segmentation import TableSegmentation
import utils

ts = TableSegmentation()
table_image = ts.img
gray = ts.gray.copy()
cropped_table_image = ts.find_and_crop_biggest_region(ts.binary)

# beginning
st.set_page_config(layout="wide")

original_col, cropped_col = st.columns(2)

with original_col:
    st.header("Original Image")
    st.text("Original")
    st.image(table_image)

with cropped_col:
    st.header("Grayscale Image")
    st.text("cropped biggest region")
    st.image(cropped_table_image)

#  =========================Segmentation result========================
overlay_image = cropped_table_image
overlay_mask = ts.pic.empty_mask(table_image)

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


cropped_mask_with_vertical_lines = ts.crop_image_with_biggest_contour(mask_with_vertical_lines.copy())
cropped_mask_with_horizontal_lines = ts.crop_image_with_biggest_contour(mask_with_horizontal_lines.copy())

#  =======================Overlay==========================
cropped_table_mask_with_horizontal_lines = ts.pic.empty_mask(cropped_table_image)
with st.container():
    st.header("Overlay")
    with st.container():
        horizontal_lines_col, vertical_lines_col = st.columns(2)

        horizontal_lines_col.markdown(
            utils.make_overlay(
                cropped_table_image, cropped_mask_with_horizontal_lines
            ),
            unsafe_allow_html=True,
        )
        vertical_lines_col.markdown(
            utils.make_overlay(cropped_table_image, cropped_mask_with_vertical_lines),
            unsafe_allow_html=True,
        )


#  =======================Todo==========================
st.header("TODO")

for i in ts.horizontal_segments():
    st.image(i)

st.header("edgeMap")
edgeMap = imutils.auto_canny(gray)
st.image(edgeMap)

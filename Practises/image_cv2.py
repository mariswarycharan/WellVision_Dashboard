import streamlit as st
import cv2
from PIL import Image
import tempfile
import numpy as np
from skimage import filters, feature, exposure

st.set_page_config(page_title="X-ray image analysis", page_icon="❄️")

hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("Well Vision images Analysis")

input_type = st.radio("Enter the type", ["Image", "Video"], horizontal=True)

file_uploader_image = st.file_uploader("Upload a content")

generate_button = st.button("Generate")

def apply_filters(image):
    filters_applied = {}
    # OpenCV Color Maps
    color_maps = {
        "Autumn": cv2.COLORMAP_AUTUMN,
        "Bone": cv2.COLORMAP_BONE,
        "Jet": cv2.COLORMAP_JET,
        "Winter": cv2.COLORMAP_WINTER,
        "Rainbow": cv2.COLORMAP_RAINBOW,
        "Ocean": cv2.COLORMAP_OCEAN,
        "Summer": cv2.COLORMAP_SUMMER,
        "Spring": cv2.COLORMAP_SPRING,
        "Cool": cv2.COLORMAP_COOL,
        "HSV": cv2.COLORMAP_HSV,
        "Pink": cv2.COLORMAP_PINK,
        "Hot": cv2.COLORMAP_HOT
    }

    for name, cmap in color_maps.items():
        filters_applied[f"Color Map - {name}"] = cv2.applyColorMap(image, cmap)
    
    # OpenCV Filters
    filters_applied["Gaussian Blur"] = cv2.GaussianBlur(image, (5, 5), 0)
    filters_applied["Median Blur"] = cv2.medianBlur(image, 5)
    filters_applied["Bilateral Filter"] = cv2.bilateralFilter(image, 9, 75, 75)
    filters_applied["Canny Edge"] = cv2.Canny(image, 100, 200)
    
    # skimage Filters
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filters_applied["Sobel Filter"] = filters.sobel(img_gray)
    filters_applied["Prewitt Filter"] = filters.prewitt(img_gray)
    filters_applied["Scharr Filter"] = filters.scharr(img_gray)
    filters_applied["Roberts Filter"] = filters.roberts(img_gray)
    filters_applied["Unsharp Mask"] = filters.unsharp_mask(img_gray, radius=1, amount=1)
    filters_applied["Local Binary Pattern"] = feature.local_binary_pattern(img_gray, P=8, R=1, method="uniform")

    # Image Adjustment with skimage
    filters_applied["Histogram Equalization"] = exposure.equalize_hist(img_gray)

    return filters_applied

# Saves
if generate_button:
    if input_type == "Image":
        c1, c2 = st.columns(2)
        img = Image.open(file_uploader_image)
        img.save("img.jpg")
        # OpenCv Read
        img = cv2.imread("img.jpg")
        
        filters_applied = apply_filters(img)
        
        half = len(filters_applied) // 2
        with c1:
            for i, (name, filtered_img) in enumerate(filters_applied.items()):
                if i < half:
                    st.image(filtered_img, caption=name, use_column_width=True, clamp=True)
        with c2:
            for i, (name, filtered_img) in enumerate(filters_applied.items()):
                if i >= half:
                    st.image(filtered_img, caption=name, use_column_width=True, clamp=True)
    
    if input_type == "Video":
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(file_uploader_image.read())
        
        cap = cv2.VideoCapture(tfile.name)
            
        st.subheader("Raw Image")
        
        raw_img_show = st.image([])
        
        c1, c2 = st.columns(2)
        
        img_placeholders = {}
        half = len(apply_filters(np.zeros((1, 1, 3), np.uint8))) // 2
        with c1:    
            for i in range(half):
                img_placeholders[f"im{i+1}_show"] = st.image([])
        with c2:    
            for i in range(half, len(apply_filters(np.zeros((1, 1, 3), np.uint8)))):
                img_placeholders[f"im{i+1}_show"] = st.image([])
        
        while cap.isOpened():
            ret, img = cap.read()
            
            raw_img_show.image(img, caption="Raw Image", use_column_width=True, clamp=True)
            
            if not ret:
                break

            filters_applied = apply_filters(img)
            for i, (name, filtered_img) in enumerate(filters_applied.items()):
                img_placeholders[f"im{i+1}_show"].image(filtered_img, caption=name, use_column_width=True, clamp=True)

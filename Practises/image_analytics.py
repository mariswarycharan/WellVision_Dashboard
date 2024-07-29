import streamlit as st
import cv2
from reportlab.platypus import Image
import tempfile


st.set_page_config(page_title="X-ray image analysis",page_icon="❄️")

hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)




st.title("Well Vision images Analysis")

input_type = st.radio("Enter the type",["Image","Video"],horizontal=True)

file_uploader_image  = st.file_uploader("Upload a content")
        
generate_button = st.button("Generate")
# Saves
if generate_button:
    
    
    if input_type == "Image":
        from PIL import Image
        
        
        c1,c2= st.columns(2)
        img = Image.open(file_uploader_image)
        img = img.save("img.jpg")
        # OpenCv Read
        img = cv2.imread("img.jpg")
        
        st.subheader("Raw Image")
        st.image(img, use_column_width=True)
        
        im1 = cv2.applyColorMap(img, cv2.COLORMAP_AUTUMN)
        im2 = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
        im3 = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        im4 = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
        im5 = cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW)
        im6 = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
        im7 = cv2.applyColorMap(img, cv2.COLORMAP_SUMMER)
        im8 = cv2.applyColorMap(img, cv2.COLORMAP_SPRING)
        im9 = cv2.applyColorMap(img, cv2.COLORMAP_COOL)
        im10 = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
        im11 = cv2.applyColorMap(img, cv2.COLORMAP_PINK)
        im12 = cv2.applyColorMap(img, cv2.COLORMAP_HOT)

        with c1:
            st.image(im1,use_column_width=True)
            st.image(im2,use_column_width=True)
            st.image(im3,use_column_width=True)
            st.image(im4,use_column_width=True)
            st.image(im5,use_column_width=True)
            st.image(im6,use_column_width=True)
            
        with c2:
            st.image(im7,use_column_width=True)
            st.image(im8,use_column_width=True)
            st.image(im9,use_column_width=True)
            st.image(im10,use_column_width=True)
            st.image(im11,use_column_width=True)
            st.image(im12,use_column_width=True)
    
    

    if input_type == "Video":
        
        
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(file_uploader_image.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st.subheader("Raw Image")
        
        raw_img_show = st.image([]) 
        c1,c2= st.columns(2)
        
        with c1:    
            im1_show = st.image([])
            im2_show = st.image([])
            im3_show = st.image([])
            im4_show = st.image([])
            im5_show = st.image([])
            im6_show = st.image([])
        with c2:    
            im7_show = st.image([])
            im8_show = st.image([])
            im9_show = st.image([])
            im10_show = st.image([])
            im11_show = st.image([])
            im12_show = st.image([])
        
        while cap.isOpened():
            
            ret, img = cap.read()
            
            
            raw_img_show.image(img, caption="Raw Image", use_column_width=True, clamp=True)
            
            im1 = cv2.applyColorMap(img, cv2.COLORMAP_AUTUMN)
            im2 = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
            im3 = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            im4 = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
            im5 = cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW)
            im6 = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
            im7 = cv2.applyColorMap(img, cv2.COLORMAP_SUMMER)
            im8 = cv2.applyColorMap(img, cv2.COLORMAP_SPRING)
            im9 = cv2.applyColorMap(img, cv2.COLORMAP_COOL)
            im10 = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
            im11 = cv2.applyColorMap(img, cv2.COLORMAP_PINK)
            im12 = cv2.applyColorMap(img, cv2.COLORMAP_HOT)

            with c1:
                im1_show.image(im1,use_column_width=True)
                im2_show.image(im2,use_column_width=True)
                im3_show.image(im3,use_column_width=True)
                im4_show.image(im4,use_column_width=True)
                im5_show.image(im5,use_column_width=True)
                im6_show.image(im6,use_column_width=True)
                
            with c2:
                im7_show.image(im7,use_column_width=True)
                im8_show.image(im8,use_column_width=True)
                im9_show.image(im9,use_column_width=True)
                im10_show.image(im10,use_column_width=True)
                im11_show.image(im11,use_column_width=True)
                im12_show.image(im12,use_column_width=True)
          
        

        
        

        
        
        
        
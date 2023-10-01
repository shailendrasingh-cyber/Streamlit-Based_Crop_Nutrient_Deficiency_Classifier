import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from keras.preprocessing import image
from PIL import Image
import streamlit as st
st.set_page_config(layout="wide")

def set_bg_hack_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(ig.jpg);
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to load uploaded image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Function to check the image
@st.cache(ttl=48*3600)
def check():
    lr = keras.models.load_model('weights.hdf5')

    # Prediction Pipeline
    class Preprocessor(BaseEstimator, TransformerMixin):
        def fit(self, img_object):
            return self

        def transform(self, img_object):
            img_array = image.img_to_array(img_object)
            expanded = (np.expand_dims(img_array, axis=0))
            return expanded

    class Predictor(BaseEstimator, TransformerMixin):
        def fit(self, img_array):
            return self

        def predict(self, img_array):
            probabilities = lr.predict(img_array)
            predicted_class = ['P_Deficiency', 'Healthy', 'N_Deficiency', 'K_Deficiency'][probabilities.argmax()]
            return predicted_class

    full_pipeline = Pipeline([('preprocessor', Preprocessor()),
                              ('predictor', Predictor())])
    return full_pipeline

# Function to display output
def output(full_pipeline, img):
    img = img.resize((224, 224))
    prediction = full_pipeline.predict(img)
    return prediction

# Function to display nutrient deficiency explanations as collapsible panels
def display_deficiency_explanations():
    st.sidebar.title("Options")
    
    option = st.sidebar.selectbox("Select an Option", ["Home", "Prediction", "What to do"])
    
    if option == "Home":
        st.markdown("<h1 style='text-align: center; color: purple;'>NutriScan: Crop Health Insights</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: purple;'>Image Classification Using CNN for identifying Plant Nutrient Deficiencies</h2>", unsafe_allow_html=True)

        st.image("ig.jpg" , use_column_width=True)

        st.markdown("<p style='text-align: center; color: black;'>Made by <a href='https://ssinghportfolio.netlify.app/' target='_blank'>Shailendra Singh</a></p>", unsafe_allow_html=True)

        set_bg_hack_url()
    
    elif option == "Prediction":
        image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
        
        prediction = ''
        
        if st.button('Predict'):
            if image_file is not None:
                with st.spinner('Loading Image and Model...'):
                    full_pipeline = check()
                file_details = {"filename": image_file.name, "filetype": image_file.type, "filesize": image_file.size}
                st.write(file_details)
                img = load_image(image_file)
                w = img.size[0]
                h = img.size[1]
                if w > h:
                    w = 600
                    st.image(img, width=w)
                else:
                    w = w * (600.0 / h)
                    st.image(img, width=int(w))
                with st.spinner('Predicting...'):
                    prediction = output(full_pipeline, img)
                st.success(prediction)
    
    elif option == "What to do":
        st.markdown("<h2 style='text-align: center; color: black;'>Deficiency Explanations</h2>", unsafe_allow_html=True)
        with st.expander("P_Deficiency"):
            st.write("पौधों की फॉस्फोरस कमी को दूर करने के लिए, पीएच समायोजन, फास्फोरस युक्त पोषण, सही तापमान, और अनुपात का ध्यान रखने के साथ पौधों की स्वस्थता को सुनिश्चित करने के लिए सुपरफॉस्फेट, बोन मील, रॉक फॉस्फेट, और ट्रिपल सुपरफॉस्फेट जैसे फॉस्फोरस-युक्त उर्वरकों का उपयोग करें।")

        with st.expander("N_Deficiency"):
            st.write("अगर पौधों में अजेयक्त नाइट्रोजन (N) की कमी हो, तो पौधों की स्वस्थता को सुनिश्चित करने के लिए निम्नलिखित कदम उठाएं: पीएच समायोजन, नाइट्रोजन-युक्त पोषण, सही तापमान, और अनुपात के साथ सुपरफोस्फेट, यूरिया, निट्रेट नाइट्रोजन, और गुआनो पोषणीय उर्वरकों का उपयोग करें।")

        with st.expander("K_Deficiency"):
            st.write("जब पौधों में पोटैशियम (K) की कमी हो, तो पौधों की स्वस्थता को सुनिश्चित करने के लिए निम्नलिखित कदम उठाएं: पीएच समायोजन, पोटैशियम-युक्त पोषण, सही तापमान, और अनुपात के साथ पोटैशियम सल्फेट, पोटैशियम नाइट्रेट, और वर्मी कम्पोस्ट जैसे पोटैशियम-युक्त उर्वरकों का उपयोग करें।")

        with st.expander("Healthy"):
            st.write("आपकी पौधों की सेहतबख्शियों को देखकर यह साबित होता है कि आप एक माहिर किसान हैं, और आपके प्यार और मेहनत से पौधों को स्वस्थ रखने का परिणाम यह है। आपका सफल होने का सबूत!.")

def main():
    display_deficiency_explanations()

if __name__ == '__main__':
    main()

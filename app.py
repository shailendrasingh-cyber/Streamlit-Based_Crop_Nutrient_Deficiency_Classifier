import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tensorflow import keras
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from keras.preprocessing import image
from PIL import Image
import streamlit as st

st.set_page_config(layout="wide")

# Function to initialize the model and prediction pipeline
@st.cache_resource
def initialize_pipeline():
    model = keras.models.load_model('weights.hdf5', compile=False)
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(reduction='sum_over_batch_size'), metrics=['accuracy'])

    class Preprocessor(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            img_array = image.img_to_array(X)
            return np.expand_dims(img_array, axis=0)

    class Predictor(BaseEstimator, TransformerMixin):
        def __init__(self, model):
            self.model = model

        def fit(self, X, y=None):
            return self

        def predict(self, X):
            probabilities = self.model.predict(X)
            classes = ['P_Deficiency', 'Healthy', 'N_Deficiency', 'K_Deficiency']
            return classes[np.argmax(probabilities)]

    return Pipeline([
        ('preprocessor', Preprocessor()),
        ('predictor', Predictor(model))
    ])

# Function to process and make predictions
def make_prediction(pipeline, img):
    img = img.resize((224, 224))
    prediction = pipeline.predict(img)
    return prediction

# Main function
def main():
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Select an Option", ["Home", "Prediction", "What to do"])

    if option == "Home":
        st.markdown("<h1 style='text-align: center; color: purple;'>NutriScan: Crop Health Insights</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: purple;'>Image Classification Using CNN for Identifying Plant Nutrient Deficiencies</h2>", unsafe_allow_html=True)
        st.image("ig.jpg", use_column_width=True)
    
    elif option == "Prediction":
        st.markdown("<h3 style='text-align: center;'>Upload an Image or Click a Photo</h3>", unsafe_allow_html=True)

        # Camera input
        image_file = st.camera_input("Take a photo")  # Camera input for direct photo capture
        
        # File uploader as fallback
        if not image_file:
            image_file = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])

        if image_file is not None:
            img = Image.open(image_file)
            st.image(img, caption="Captured/Uploaded Image", use_column_width=True)

            if st.button("Predict"):
                with st.spinner('Loading Model and Predicting...'):
                    pipeline = initialize_pipeline()
                    prediction = make_prediction(pipeline, img)
                st.success(f"Prediction: {prediction}")

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


if __name__ == '__main__':
    main()

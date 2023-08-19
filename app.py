import streamlit as st
import tensorflow as tf
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model("skin_cancer_detection_model.h5")

# Streamlit UI
st.title("Skin Cancer Detection")
st.markdown(
    """ 
    ##### Skin cancer is a common type of cancer that can usually be cured if detected early. 
    ##### This tool uses a deep learning model to predict whether a skin lesion is malignant (cancerous) or benign (non-cancerous).
    ##### Upload an image to get a prediction.
    """
)

# Upload and process image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

class_labels = ["Benign", "Malignant"]  # Define class labels for mapping

show_prediction = False  # Initialize the flag to show prediction

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = Image.open(uploaded_file)
    img = img.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)

    # Make prediction when "Predict" button is clicked
    if st.button("Predict", key="predict_button"):
        show_prediction = True
        with st.spinner("Predicting..."):
            # Make prediction and show result
            prediction = model.predict(img_array)
            predicted_class = class_labels[int(prediction[0] >= 0.5)]
            confidence = prediction[0][0]
            st.write("### <span style='color: green;'>Prediction complete", unsafe_allow_html=True)

    # Display prediction result if the prediction has been made
    if show_prediction:
        if predicted_class == "Benign" :
            st.write(f"##### Skin is showing no sign of cancer with a confidence level of {confidence:.2%}")
        else:
            st.write(f"##### Skin has cancerous traits with a confidence level of {confidence:.2%}")
        st.write(f"##### The model predicts that the lesion is: **{predicted_class}**")
        st.write('##### For more information visit https://www.nhs.uk/conditions/non-melanoma-skin-cancer/')

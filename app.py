import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Pneumonia Detection App",
    page_icon="🩺",
    layout="centered"
)

# Load Serialized Model (Cached for performance)
@st.cache_resource
def load_model():
    model_path = "best_pneumonia_transfer_model.keras"
    return tf.keras.models.load_model(model_path)

model = load_model()
class_labels = ['Normal', 'Lung Opacity', 'No Lung Opacity / Not Normal']

st.title("🩺 Chest X-Ray Pneumonia Detection")
st.write("Upload a chest X-ray image (DICOM or standard image formats like PNG/JPG) to classify the condition using your transfer learning model.")

# File Uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png", "dcm"])

if uploaded_file is not None:
    # Check if the uploaded file is a DICOM file
    if uploaded_file.name.endswith('.dcm'):
        # Read the DICOM file using pydicom
        dicom_data = pydicom.dcmread(uploaded_file)
        image_array = dicom_data.pixel_array
        
        # Normalize the pixel array to 0-255 and convert to PIL Image
        image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)
        image = Image.fromarray(image_array).convert('L')
    else:
        # Standard image formats (PNG, JPG)
        image = Image.open(uploaded_file).convert('L')
        
    st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)
    
    if st.button("Run Prediction"):
        with st.spinner("Analyzing image..."):
            # Preprocessing matching your training pipeline
            raw_img = np.array(image, dtype=np.float32)
            img_min, img_max = raw_img.min(), raw_img.max()
            
            if img_max > img_min:
                img_normalized = (raw_img - img_min) / (img_max - img_min)
            else:
                img_normalized = np.zeros_like(raw_img)
                
            # Resize to (224, 224)
            img_resized = cv2.resize(img_normalized, (224, 224), interpolation=cv2.INTER_AREA)
            
            # Expand dims & replicate channels to match 3-channel input expectation (224, 224, 3)
            img_expanded = np.expand_dims(img_resized, axis=-1)
            img_3channel = np.repeat(img_expanded, 3, axis=-1)
            img_batch = np.expand_dims(img_3channel, axis=0)
            
            # Make Prediction
            predictions = model.predict(img_batch, verbose=0)
            pred_idx = np.argmax(predictions[0])
            predicted_class = class_labels[pred_idx]
            confidence = predictions[0][pred_idx] * 100
            
            # Display Results
            st.success(f"**Prediction:** {predicted_class}")
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
            
            # Breakdown of all class probabilities
            st.subheader("Class Probability Breakdown:")
            prob_dict = {class_labels[i]: float(predictions[0][i]) * 100 for i in range(len(class_labels))}
            st.bar_chart(prob_dict)

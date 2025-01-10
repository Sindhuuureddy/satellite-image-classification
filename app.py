import streamlit as st
import requests
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained model
MODEL_PATH = "models/crop_classification_model.h5"
model = load_model(MODEL_PATH)

# Define the categories
CATEGORIES = ["Rabi Crop", "Kharif Crop", "Other"]

def fetch_satellite_image(lat, lon, zoom=12):
    """
    Fetch satellite image using Google Maps Static API.
    """
    api_key = "YOUR_GOOGLE_MAPS_API_KEY"
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size=640x640&maptype=satellite&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(requests.get(url, stream=True).raw)
    else:
        st.error("Failed to fetch satellite image. Check API key or input coordinates.")
        return None

def preprocess_image(image):
    """
    Preprocess the image for model prediction.
    """
    image = image.resize((128, 128))  # Resize to match the model's input size
    image = np.array(image) / 255.0   # Normalize pixel values
    return np.expand_dims(image, axis=0)

def predict_category(image):
    """
    Predict the category of the satellite image.
    """
    predictions = model.predict(image)
    category = CATEGORIES[np.argmax(predictions)]
    confidence = np.max(predictions)
    return category, confidence

# Streamlit UI
st.title("Satellite Image Classification")
st.sidebar.header("Input Coordinates")

# Input latitude and longitude
latitude = st.sidebar.number_input("Enter Latitude", value=20.5937, format="%.6f")
longitude = st.sidebar.number_input("Enter Longitude", value=78.9629, format="%.6f")

# Fetch and classify image
if st.sidebar.button("Fetch and Classify"):
    st.write(f"Fetching satellite image for coordinates: ({latitude}, {longitude})...")
    image = fetch_satellite_image(latitude, longitude)

    if image:
        st.image(image, caption="Satellite Image", use_column_width=True)

        # Preprocess and classify the image
        processed_image = preprocess_image(image)
        category, confidence = predict_category(processed_image)

        # Display results
        st.subheader("Classification Result")
        st.write(f"**Category**: {category}")
        st.write(f"**Confidence**: {confidence:.2f}")

# Optional: Add a map for user to select coordinates
st.sidebar.subheader("Select Coordinates on Map")
st.map(data={"lat": [latitude], "lon": [longitude]})

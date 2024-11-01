import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image  # Ensure PIL is imported

# Load the CNN model and label binarizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load your label binarizer
with open('captcha_labels.pickle', 'rb') as f:
    lb = pickle.load(f)

# Define the character mapping based on the model's output
# characters = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'  # Update based on your training set

# Function to preprocess the uploaded image and predict CAPTCHA text
def process_captcha_image(image):
    # Convert the uploaded image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Add some extra padding around the image
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    # Apply thresholding
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find the contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letter_image_regions = []

    # Loop through each contour and extract the letters
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        
        # Check if the contour width is too wide (indicating joined letters)
        if w / h > 1.25:
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions.append((x, y, w, h))

    # Sort letter images from left to right
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Ensure we only process up to 4 letters
    letter_image_regions = letter_image_regions[:4]

    # Create an output image for visualization
    output = cv2.merge([gray] * 3)

    predictions = []

    for letter_bounding_box in letter_image_regions:
        x, y, w, h = letter_bounding_box

        # Extract the letter with a small border
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
        letter_image = cv2.resize(letter_image, (20, 20))
        
        # Reshape for model prediction
        letter_image = np.expand_dims(letter_image, axis=2)  # Add channel dimension
        letter_image = np.expand_dims(letter_image, axis=0)   # Add batch dimension

        # Make prediction
        pred = model.predict(letter_image)
        letter = lb.inverse_transform(pred)[0]  # Use your label binarizer to get the letter
        predictions.append(letter)

        # Draw the prediction on the output image
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 5, y + h + 5), (0, 255, 0), 1)
        cv2.putText(output, letter, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Combine the predicted letters to form the CAPTCHA text
    captcha_text = "".join(predictions)
    return captcha_text, output

# Streamlit app interface
st.set_page_config(page_title="CAPTCHA Reader", page_icon="🤖", layout="wide")
st.title("CleverC4PTCHA: Crack the Code!")
st.write("Upload a CAPTCHA image to get the predicted text.")

# Upload image functionality
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display loading indicator
    with st.spinner("Processing your image..."):
        image = np.array(Image.open(uploaded_file))
        captcha_text, output_image = process_captcha_image(image)
        
    st.write("### Predicted CAPTCHA text:")
    st.markdown(f"<h2 style='color: green;'>{captcha_text}</h2>", unsafe_allow_html=True)
    st.image(output_image, caption="Processed CAPTCHA", use_column_width=True)

    # Add download button
    output_path = "processed_captcha.png"
    cv2.imwrite(output_path, output_image)
    with open(output_path, "rb") as file:
        st.download_button("Download Processed Image", file, file_name="processed_captcha.png")

# Add footer
st.markdown("---")
st.write("Created with ❤️ by Priyanshu Kumar")
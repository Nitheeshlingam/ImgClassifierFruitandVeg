import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
import numpy as np
from PIL import Image

st.header('Image Classification Model')

# Load the trained model
model = load_model('E:\ImageClassification\Image_classify.keras')

# Define the category labels
data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

# Set image dimensions
img_height = 180
img_width = 180

# Drag-and-drop image uploader
uploaded_file = st.file_uploader("Drag and drop an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Resize the image for the model
    image = image.resize((img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image)  # Convert to array
    img_bat = tf.expand_dims(img_arr, 0)  # Add batch dimension

    # Predict the class
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])  # Apply softmax to the first element
    predicted_class = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100

    # Display the prediction
    st.write(f"Predicted Class: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    # (Optional) Evaluate the model using testing dataset if applicable
    st.subheader("Evaluation Metrics (for reference)")
    # Simulated test data setup (Replace this with actual test data)
    # `x_test` is a numpy array of preprocessed images.
    # `y_test` is a numpy array of one-hot encoded labels or class indices.
    x_test = np.random.random((50, img_height, img_width, 3))  # Example test data
    y_test = np.random.randint(0, len(data_cat), 50)           # Example test labels (class indices)

    # Predict classes on the test data
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')

    # Display the metrics
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1-Score: {f1:.2f}")

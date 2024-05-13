import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

# Set paths for validation
validation_csv_file_path = 'D:/CTS Project/Evaluation_Set/Validation/Processed_RFMiD_Validation_Labels.csv'
validation_image_folder_path = 'D:/CTS Project/Evaluation_Set/Validation'
image_size = (128, 128)  # Adjust the size according to your requirement

# Load the validation CSV file containing image labels
df_validation = pd.read_csv(validation_csv_file_path)

# Preprocess the validation data
X_val = []  # List to store validation image arrays
y_val = []  # List to store corresponding validation labels
for index, row in df_validation.iterrows():
    image_path = os.path.join(validation_image_folder_path, row['Image_Name'])
    img = Image.open(image_path).convert("RGB")
    img = img.resize(image_size)
    img_array = np.array(img) / 255.0
    X_val.append(img_array)
    y_val.append(row['Disease_Name'])

X_val = np.array(X_val)

# Load label encoder classes
le = LabelEncoder()
le.classes_ = np.load("label_encoder_classes1.npy", allow_pickle=True)

# Convert string labels to integer labels using label encoder
y_val = le.transform(y_val)

# Load the trained model
model = load_model("final_model1.h5")

# Evaluate the model on the validation data
loss, accuracy = model.evaluate(X_val, y_val)
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

# Predict on the validation data and display results
for i in range(len(X_val)):
    # Reshape the validation image to match the model input shape
    val_img_array = np.expand_dims(X_val[i], axis=0)

    # Make prediction on the validation image
    predicted_label_idx = np.argmax(model.predict(val_img_array), axis=-1)[0]
    predicted_disease = le.inverse_transform([predicted_label_idx])[0]

    # Display the output image along with the predicted disease name and accuracy
    plt.imshow(X_val[i])
    plt.title(f"Predicted Disease: {predicted_disease}, Accuracy: {accuracy:.2f}")
    plt.axis('off')
    plt.show()

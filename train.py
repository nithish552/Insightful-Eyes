
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Set paths and parameters
csv_file_path = 'D:/CTS Project/Training_Set/Processed_RFMiD_Training_Labels.csv'
image_folder_path = 'D:/CTS Project/Training_Set/Training'
batch_size = 32
image_size = (128, 128)  # Adjust the size according to your requirement
num_epochs = 10

# Load the CSV file containing image labels
df = pd.read_csv(csv_file_path)

# Preprocess the data and split into training and validation sets
X = []  # List to store image arrays
y = []  # List to store corresponding labels
for index, row in df.iterrows():
    image_path = os.path.join(image_folder_path, row['Image_Name'])
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img)
    X.append(img_array)
    y.append(row['Disease_Name'])

X = np.array(X)
y = np.array(y)

le = LabelEncoder()
y = le.fit_transform(y)
np.save("label_encoder_classes1.npy", le.classes_)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data normalization
X_train = X_train / 255.0
X_val = X_val / 255.0

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Load the VGG16 model with pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

# Fine-tune the last layers of VGG16
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Build the transfer learning model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(le.classes_), activation='softmax'))

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks (optional)
checkpoint = ModelCheckpoint("best_model1.h5", monitor='val_accuracy', verbose=1, save_best_only=True)

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=num_epochs,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint]
)

# Save the trained model
model.save("final_model1.h5")

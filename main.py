import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#809 pokemons
#initialisation
pokemon_df = pd.read_csv('pokemon.csv')
encoder = LabelEncoder()
print(pokemon_df)

#sanitising dataset
pokemon_df.drop('Evolution', axis = 1, inplace=True) #dropping evolution since most of it is empty
pokemon_df.dropna(inplace=True)
print(pokemon_df)
print(len(pokemon_df))

#encoding the data
for column in pokemon_df.columns:
    if column != "Name":
        pokemon_df[column] = encoder.fit_transform(pokemon_df[column])

#loading images
images  = []
labels = []
image_dir = "./images"
for idx, row in pokemon_df.iterrows():
    name = row['Name']
    image_path = os.path.join(image_dir, f"{name.lower()}.png")
    if os.path.exists(image_path):
        
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Standard size for many CNN models
        img_array = np.array(img) / 255.0  # Normalize pixel values
        
        images.append(img_array)
        labels.append(row['Name'])
    else:
        print(f"Image not found for {name}")
        
images = np.array(images)
labels = np.array(labels)

# 1. Split the data into training and testing sets

# Create a mapping of Pokemon names to numeric indices
unique_labels = np.unique(labels)
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
numeric_labels = np.array([label_to_idx[label] for label in labels])

# Convert to one-hot encoding
num_classes = len(unique_labels)
one_hot_labels = to_categorical(numeric_labels, num_classes)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    images, one_hot_labels, test_size=0.2, random_state=42)

# 2. Build and train a CNN model
model = Sequential([
    # Convolutional layers
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Flatten and dense layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Add dropout to prevent overfitting
    Dense(num_classes, activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,  # You might need more epochs
    batch_size=32,
    validation_split=0.2
)

# 3. Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Save the model
model.save('pokemon_classifier.h5')

# Create a function to make predictions on new images
def predict_pokemon(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    prediction = model.predict(img_array)
    predicted_idx = np.argmax(prediction[0])
    predicted_pokemon = unique_labels[predicted_idx]
    confidence = prediction[0][predicted_idx]
    
    return predicted_pokemon, confidence
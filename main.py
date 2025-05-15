import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def load_pokemon_data(image_dir):
    images = []
    labels = []
    
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            # Load image
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = cv2.resize(img, (128, 128))  # Resize to consistent dimensions
            
            # Extract label from filename (format: "pokemonname_number.png")
            label = filename.split('_')[0]
            
            images.append(img)
            labels.append(label)
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    images_normalized = images.astype('float32') / 255.0
    
    return images_normalized, labels

def create_sequential_pokemon_model(input_shape=(128, 128, 3), num_classes=809):
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Conv Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third Conv Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Fourth Conv Block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Classifier Head
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_pokemon_classes():
    # This function should return the list of Pokémon classes in the order they were encoded
    # If you have this information saved somewhere, load it from there
    # Otherwise, you'll need to recreate it by loading and encoding your training data
    
    image_dir = './augmented_images'  # Update this path if needed
    _, labels = load_pokemon_data(image_dir)
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    return label_encoder.classes_

def predict_pokemon(image_path, top_k=1):
    model = load_model('sequential_pokemon_classifier.tflite')
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0
    
    predictions = model.predict(img)[0]
    
    # Get ordered list of class names (Pokémon names)
    pokemon_classes = get_pokemon_classes()
    # Find top-k predictions
    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_probabilities = predictions[top_indices]
    top_pokemon_names = [pokemon_classes[i] for i in top_indices]
    
    # Create a formatted output
    results = []
    for i, (name, prob) in enumerate(zip(top_pokemon_names, top_probabilities)):
        results.append({
            "rank": i+1,
            "pokemon": name,
            "probability": float(prob),
            "confidence": f"{prob*100:.2f}%"
        })
    
    return {
        "top_prediction": results[0]["pokemon"],
        "confidence": results[0]["confidence"],
        "all_results": results
    }

def predict_pokemon_tflite(image_path, tflite_model_path = "./sequential_pokemon_classifier.tflite", top_k=1):
    """
    Make predictions using the TFLite model
    
    Args:
        image_path: Path to the image to predict
        tflite_model_path: Path to the TFLite model
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with prediction results
    """
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0).astype('float32') / 255.0
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Get ordered list of class names (Pokémon names)
    pokemon_classes = get_pokemon_classes()
    
    # Find top-k predictions
    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_probabilities = predictions[top_indices]
    top_pokemon_names = [pokemon_classes[i] for i in top_indices]
    
    # Create a formatted output
    results = []
    for i, (name, prob) in enumerate(zip(top_pokemon_names, top_probabilities)):
        results.append({
            "rank": i+1,
            "pokemon": name,
            "probability": float(prob),
            "confidence": f"{prob*100:.2f}%"
        })
    
    return {
        "top_prediction": results[0]["pokemon"],
        "confidence": results[0]["confidence"],
        "all_results": results
    }

if __name__ == "__main__":
    # # Load your images
    # image_dir = './augmented_images'  # Update this path
    # X, y = load_pokemon_data(image_dir)
    # print("Num_pokemon:" + str(len(y)))
    # Encode labels
    #label_encoder = LabelEncoder()
    # y_encoded = label_encoder.fit_transform(y)
    # y_categorical = to_categorical(y_encoded)

    # # Split data into training and validation sets
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y_categorical, test_size=0.2, random_state=42
    # )
    # model = create_sequential_pokemon_model(
    # input_shape=X_train.shape[1:], 
    # num_classes=len(label_encoder.classes_)
    # )

    # # 2. Train model
    # history = model.fit(
    #     X_train, y_train,
    #     validation_data=(X_val, y_val),
    #     epochs=30,  # Start with 30 epochs, can increase if needed
    #     batch_size=32
    # )

    # # 3. Save model
    # model.save('sequential_pokemon_classifier.h5')
    #print(predict_pokemon_tflite("./images/abra.png"))
    try:
        from app import app
        app.run(debug=False)
    except Exception as e:
        print(e)
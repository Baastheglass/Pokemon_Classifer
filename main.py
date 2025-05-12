# import os
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# import numpy as np
# from PIL import Image
# from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.utils.all_utils import to_categorical
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
# import matplotlib.pyplot as plt
# # 809 pokemons
# # initialisation
# pokemon_df = pd.read_csv('pokemon.csv')
# encoder = LabelEncoder()
# print(pokemon_df)

# #sanitising dataset
# pokemon_df.drop('Evolution', axis = 1, inplace=True) #dropping evolution since most of it is empty
# pokemon_df.dropna(inplace=True)
# print(pokemon_df)
# print(len(pokemon_df))

# #encoding the data
# for column in pokemon_df.columns:
#     if column != "Name":
#         pokemon_df[column] = encoder.fit_transform(pokemon_df[column])

# #loading images
# images  = []
# labels = []
# image_dir = "./images"
# for idx, row in pokemon_df.iterrows():
#     name = row['Name']
#     image_path = os.path.join(image_dir, f"{name.lower()}.png")
#     if os.path.exists(image_path):
        
#         img = Image.open(image_path)
#         img = img.resize((224, 224))  # Standard size for many CNN models
#         img_array = np.array(img) / 255.0  # Normalize pixel values
        
#         images.append(img_array)
#         labels.append(row['Name'])
#     else:
#         print(f"Image not found for {name}")
        
# images = np.array(images)
# labels = np.array(labels)

# #Create a mapping of Pokemon names to numeric indices
# unique_labels = np.unique(labels)
# label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
# numeric_labels = np.array([label_to_idx[label] for label in labels])

# #Convert to one-hot encoding
# num_classes = len(unique_labels)
# one_hot_labels = to_categorical(numeric_labels, num_classes)

# #Split the data
# X_train, X_test, y_train, y_test = train_test_split(
#     images, one_hot_labels, test_size=0.2, random_state=42)

# # Build and train a CNN model
# model = Sequential([
#     #Convolutional layers
#     Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
    
#     #Flatten and dense layers
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),  # Add dropout to prevent overfitting
#     Dense(num_classes, activation='softmax')
#     ])

# #Compile the model
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# #Print model summary
# model.summary()

# #Train the model
# history = model.fit(
#     X_train, y_train,
#     epochs=10,  # You might need more epochs
#     batch_size=32,
#     validation_split=0.2
# )

# # Evaluate the model
# test_loss, test_acc = model.evaluate(X_test, y_test)
# print(f"Test accuracy: {test_acc:.4f}")

# #Save the model
# model.save('pokemon_classifier.h5')

# #Create a function to make predictions on new images
# def predict_pokemon(model, image_path):
#     img = Image.open(image_path)
#     img = img.resize((224, 224))
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
#     prediction = model.predict(img_array)
#     predicted_idx = np.argmax(prediction[0])
#     predicted_pokemon = unique_labels[predicted_idx]
#     confidence = prediction[0][predicted_idx]
    
#     return predicted_pokemon, confidence
# import os
# import json
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from PIL import Image
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Dense, GlobalAveragePooling2D, Dropout, 
#     Input, BatchNormalization, LeakyReLU
# )
# from tensorflow.keras.optimizers import AdamW
# from tensorflow.keras.callbacks import (
#     ReduceLROnPlateau, EarlyStopping, 
#     ModelCheckpoint, CSVLogger
# )
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.utils import to_categorical

# class PokemonClassifier:
#     def __init__(self, csv_path, image_dir):
#         self.csv_path = csv_path
#         self.image_dir = image_dir
#         self.model = None
#         self.label_encoder = LabelEncoder()
#         self.label_mapping = {}
        
#     def preprocess_data(self, verbose=True):
#         pokemon_df = pd.read_csv(self.csv_path)
        
#         # Clean data
#         pokemon_df.drop(['Evolution'], axis=1, inplace=True, errors='ignore')
#         pokemon_df.dropna(inplace=True)
        
#         # Encode categorical features
#         for column in pokemon_df.columns:
#             if column != "Name":
#                 pokemon_df[column] = self.label_encoder.fit_transform(pokemon_df[column])
        
#         images = []
#         labels = []
#         skipped_images = []
        
#         # Load images
#         for idx, row in pokemon_df.iterrows():
#             name = row['Name'].lower()
#             image_path = os.path.join(self.image_dir, f"{name}.png")
            
#             try:
#                 if os.path.exists(image_path):
#                     img = Image.open(image_path).convert('RGB')
#                     img = img.resize((224, 224))
#                     img_array = np.array(img) / 255.0
                    
#                     images.append(img_array)
#                     labels.append(name)
#                 else:
#                     skipped_images.append(name)
#             except Exception as e:
#                 print(f"Error processing {name}: {e}")
#                 skipped_images.append(name)
        
#         images = np.array(images)
#         labels = np.array(labels)
        
#         # Create label mapping
#         unique_labels = np.unique(labels)
#         self.label_mapping = {
#             label: idx for idx, label in enumerate(unique_labels)
#         }
        
#         numeric_labels = np.array([self.label_mapping[label] for label in labels])
#         one_hot_labels = to_categorical(numeric_labels)
        
#         if verbose:
#             print(f"Total images processed: {len(images)}")
#             print(f"Unique Pokemon classes: {len(unique_labels)}")
#             print(f"Skipped images: {len(skipped_images)}")
#             if skipped_images:
#                 print("First 10 skipped Pokemon:", skipped_images[:10])
        
#         return images, one_hot_labels

#     def create_model(self, num_classes):
#         # Using a smaller EfficientNetB0 instead of B7 for faster training
#         base = EfficientNetB0(
#             weights='imagenet', 
#             include_top=False, 
#             input_shape=(224, 224, 3)
#         )
        
#         # Freeze the base model
#         base.trainable = False
        
#         inputs = Input(shape=(224, 224, 3))
        
#         x = base(inputs, training=False)
        
#         x = GlobalAveragePooling2D()(x)
#         x = BatchNormalization()(x)
        
#         # Simplified architecture
#         x = Dense(256, kernel_regularizer=l2(0.001))(x)
#         x = LeakyReLU(alpha=0.1)(x)
#         x = Dropout(0.5)(x)
        
#         outputs = Dense(
#             num_classes, 
#             activation='softmax', 
#             kernel_regularizer=l2(0.001)
#         )(x)
        
#         model = Model(inputs=inputs, outputs=outputs)
        
#         model.compile(
#             optimizer=AdamW(learning_rate=1e-4),
#             loss='categorical_crossentropy',
#             metrics=['accuracy', 'top_k_categorical_accuracy']
#         )
        
#         return model

#     def train_model(self, images, labels, epochs=30, batch_size=16):
#         # Modified to work with single instance per class
#         # Simple random split without stratification
#         X_train, X_test, y_train, y_test = train_test_split(
#             images, labels, test_size=0.2, random_state=42
#         )
        
#         # Data augmentation is crucial when having only one sample per class
#         datagen = ImageDataGenerator(
#             rotation_range=30,
#             width_shift_range=0.2,
#             height_shift_range=0.2,
#             shear_range=0.2,
#             zoom_range=0.3,
#             horizontal_flip=True,
#             vertical_flip=False,
#             brightness_range=[0.8, 1.2],
#             fill_mode='nearest'
#         )
        
#         # Callbacks
#         reduce_lr = ReduceLROnPlateau(
#             monitor='val_loss', 
#             factor=0.5, 
#             patience=3, 
#             min_lr=1e-6,
#             verbose=1
#         )
        
#         early_stopping = EarlyStopping(
#             monitor='val_accuracy', 
#             patience=10, 
#             restore_best_weights=True,
#             verbose=1
#         )
        
#         model_checkpoint = ModelCheckpoint(
#             'best_pokemon_model.h5', 
#             save_best_only=True, 
#             monitor='val_accuracy'
#         )
        
#         csv_logger = CSVLogger('training_log.csv')
        
#         # Fit the model using the generator directly
#         # Since we can't use validation_split with our small dataset
#         history = self.model.fit(
#             datagen.flow(X_train, y_train, batch_size=batch_size),
#             steps_per_epoch=max(1, len(X_train) // batch_size),
#             epochs=epochs,
#             validation_data=(X_test, y_test),
#             callbacks=[reduce_lr, early_stopping, model_checkpoint, csv_logger]
#         )
        
#         test_results = self.model.evaluate(X_test, y_test)
#         print(f"\nTest Accuracy: {test_results[1]:.4f}")
#         print(f"Top-3 Accuracy: {test_results[2]:.4f}")
        
#         self._plot_training_history(history)
        
#         return history.history

#     def _plot_training_history(self, history):
#         plt.figure(figsize=(12, 4))
        
#         plt.subplot(1, 2, 1)
#         plt.plot(history.history['accuracy'], label='Training Accuracy')
#         plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#         plt.title('Model Accuracy')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.legend()
        
#         plt.subplot(1, 2, 2)
#         plt.plot(history.history['loss'], label='Training Loss')
#         plt.plot(history.history['val_loss'], label='Validation Loss')
#         plt.title('Model Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
        
#         plt.tight_layout()
#         plt.savefig('training_metrics.png')
#         plt.close()

#     def predict(self, image_path):
#         img = Image.open(image_path).convert('RGB')
#         img = img.resize((224, 224))
#         img_array = np.array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)
        
#         prediction = self.model.predict(img_array)
#         predicted_idx = np.argmax(prediction[0])
#         confidence = prediction[0][predicted_idx]
        
#         idx_to_label = {v: k for k, v in self.label_mapping.items()}
#         predicted_pokemon = idx_to_label[predicted_idx]
        
#         # Return top 3 predictions
#         top_indices = np.argsort(prediction[0])[-3:][::-1]
#         top_predictions = [(idx_to_label[idx], prediction[0][idx]) for idx in top_indices]
        
#         return predicted_pokemon, confidence, top_predictions

#     def save_model_metadata(self):
#         metadata = {
#             'label_mapping': self.label_mapping,
#             'num_classes': len(self.label_mapping)
#         }
        
#         with open('pokemon_model_metadata.json', 'w') as f:
#             json.dump(metadata, f)

#     def run_full_pipeline(self):
#         print("Starting data preprocessing...")
#         images, labels = self.preprocess_data()
        
#         print(f"\nCreating model for {labels.shape[1]} Pokemon classes...")
#         self.model = self.create_model(num_classes=labels.shape[1])
#         self.model.summary()
        
#         print("\nTraining model...")
#         history = self.train_model(images, labels)
        
#         print("\nSaving model metadata...")
#         self.save_model_metadata()
        
#         return history

# if __name__ == "__main__":
#     classifier = PokemonClassifier(
#         csv_path='pokemon.csv',
#         image_dir='./images'
#     )
    
#     try:
#         classifier.run_full_pipeline()
        
#         # Test prediction
#         test_image = './images/pikachu.png'
#         if os.path.exists(test_image):
#             predicted_pokemon, confidence, top3 = classifier.predict(test_image)
#             print(f"\nPredicted Pokemon: {predicted_pokemon}")
#             print(f"Confidence: {confidence:.2%}")
#             print("\nTop 3 predictions:")
#             for pokemon, conf in top3:
#                 print(f"- {pokemon}: {conf:.2%}")
#         else:
#             print(f"\nTest image not found: {test_image}")
#             print("Try another Pokemon image for prediction")
#     except Exception as e:
#         import traceback
#         print(f"Error occurred: {e}")
#         traceback.print_exc()
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2L, ResNet152V2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, 
    Input, BatchNormalization, LeakyReLU
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, 
    ModelCheckpoint, CSVLogger
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import cv2
import albumentations as A

class PokemonClassifier:
    def __init__(self, csv_path, image_dir):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.model = None
        self.label_encoder = LabelEncoder()
        self.label_mapping = {}
        # Create directory for augmented images
        self.augmented_dir = "./augmented_images"
        os.makedirs(self.augmented_dir, exist_ok=True)
        
    def create_augmentations(self, image, name, num_augmentations=15):
        """Create multiple augmented versions of a single image"""
        augmented_images = []
        augmented_labels = []
        
        try:
            # Define a simpler augmentation pipeline with fewer transforms
            # Use only the most reliable, commonly available transforms
            simple_aug_pipeline = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5)
            ])
            
            # Create original + augmented versions
            for i in range(num_augmentations):
                if i == 0:
                    # Include the original image
                    aug_img = image.copy()
                else:
                    # Apply augmentation
                    aug_img = simple_aug_pipeline(image=image)['image']
                
                # Save augmented image to disk for inspection
                aug_path = os.path.join(self.augmented_dir, f"{name}_{i}.png")
                cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                
                # Normalize and add to dataset
                aug_img = aug_img / 255.0
                augmented_images.append(aug_img)
                augmented_labels.append(name)
                
            return augmented_images, augmented_labels
            
        except Exception as e:
            print(f"Error during augmentation of {name}: {e}")
            # Return just the original image as fallback
            return [image / 255.0], [name]
        
    def preprocess_data(self, verbose=True, augment=True, num_augmentations=15):
        pokemon_df = pd.read_csv(self.csv_path)
        
        # Clean data
        pokemon_df.drop(['Evolution'], axis=1, inplace=True, errors='ignore')
        pokemon_df.dropna(inplace=True)
        
        # Encode categorical features
        for column in pokemon_df.columns:
            if column != "Name":
                pokemon_df[column] = self.label_encoder.fit_transform(pokemon_df[column])
        
        images = []
        labels = []
        skipped_images = []
        processed_count = 0
        
        # Print information about the CSV file
        if verbose:
            print(f"CSV file contains {len(pokemon_df)} Pokemon entries")
            print(f"First few Pokemon names: {pokemon_df['Name'].head().tolist()}")
        
        # Load images and create augmentations
        for idx, row in pokemon_df.iterrows():
            name = row['Name'].lower()
            image_path = os.path.join(self.image_dir, f"{name}.png")
            
            if verbose and idx < 5:  # Print paths for the first few Pokemon to help with debugging
                print(f"Looking for image: {image_path}")
            
            try:
                if os.path.exists(image_path):
                    # Load image
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Warning: Could not read image {image_path} (file may be corrupted)")
                        skipped_images.append(name)
                        continue
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    
                    if augment:
                        # Use a try-except block specifically for augmentations
                        try:
                            # Create augmented versions
                            aug_images, aug_labels = self.create_augmentations(img, name, num_augmentations)
                            images.extend(aug_images)
                            labels.extend(aug_labels)
                        except Exception as e:
                            print(f"Warning: Augmentation failed for {name}, using original image only: {e}")
                            # Fall back to using just the original image if augmentation fails
                            images.append(img / 255.0)
                            labels.append(name)
                    else:
                        # Just use original image
                        images.append(img / 255.0)
                        labels.append(name)
                    
                    processed_count += 1
                    if verbose and processed_count % 10 == 0:
                        print(f"Processed {processed_count} Pokemon images so far...")
                else:
                    skipped_images.append(name)
                    if verbose and idx < 20:  # Only show first few missing images
                        print(f"Image not found: {image_path}")
            except Exception as e:
                print(f"Error processing {name}: {e}")
                skipped_images.append(name)
        
        # Check if we have any processed images
        if processed_count == 0:
            raise ValueError("No images were successfully processed. Check your image directory path and image names.")
        
        images = np.array(images)
        labels = np.array(labels)
        
        # Create label mapping
        unique_labels = np.unique(labels)
        self.label_mapping = {
            label: idx for idx, label in enumerate(unique_labels)
        }
        
        # Convert string labels to numeric indices
        numeric_labels = np.array([self.label_mapping[label] for label in labels])
        
        # Add check to ensure numeric_labels is not empty
        if len(numeric_labels) == 0:
            raise ValueError("No labels were processed. Check your data and image paths.")
        
        one_hot_labels = to_categorical(numeric_labels)
        
        if verbose:
            print(f"Total images processed: {len(images)}")
            print(f"Unique Pokemon classes: {len(unique_labels)}")
            print(f"Augmented images per class: {num_augmentations if augment else 'None'}")
            print(f"Skipped images: {len(skipped_images)}")
            if skipped_images:
                print("First 10 skipped Pokemon:", skipped_images[:10])
        
        return images, one_hot_labels, numeric_labels

    def create_model(self, num_classes, model_type='efficientnetv2'):
        # Select base model based on parameter
        if model_type == 'efficientnetv2':
            base = EfficientNetV2L(
                weights='imagenet', 
                include_top=False, 
                input_shape=(224, 224, 3)
            )
        elif model_type == 'resnet':
            base = ResNet152V2(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # First train with base model frozen
        base.trainable = False
        
        inputs = Input(shape=(224, 224, 3))
        
        x = base(inputs, training=False)
        
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        
        # More complex architecture for better feature extraction
        x = Dense(512, kernel_regularizer=l2(0.0005))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(256, kernel_regularizer=l2(0.0005))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        outputs = Dense(
            num_classes, 
            activation='softmax', 
            kernel_regularizer=l2(0.0005)
        )(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=AdamW(learning_rate=5e-5, weight_decay=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        return model

    def fine_tune_model(self, model, learning_rate=1e-5):
        """Unfreeze the base model for fine-tuning"""
        # Unfreeze the base model
        for layer in model.layers:
            if hasattr(layer, 'layers'):  # This is our base model
                # Unfreeze only the last 50 layers
                for i, l in enumerate(layer.layers):
                    if i >= len(layer.layers) - 50:  # Unfreeze last 50 layers
                        l.trainable = True
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=AdamW(learning_rate=learning_rate, weight_decay=1e-6),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        return model

    def train_model(self, images, labels, numeric_labels, epochs=50, batch_size=16, k_folds=5):
        """Train with k-fold cross-validation for better generalization"""
        # Setup k-fold cross validation
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        fold_histories = []
        fold_scores = []
        best_score = 0
        best_model_path = 'best_pokemon_model.h5'
        
        # K-fold cross validation
        for fold, (train_idx, val_idx) in enumerate(kfold.split(images, numeric_labels)):
            print(f"\n---- Training Fold {fold+1}/{k_folds} ----")
            
            X_train, X_val = images[train_idx], images[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # Recreate model for each fold
            self.model = self.create_model(num_classes=labels.shape[1], model_type='efficientnetv2')
            
            # Data augmentation
            train_datagen = ImageDataGenerator(
                horizontal_flip=True,
                rotation_range=10,
                zoom_range=0.1,
                width_shift_range=0.1,
                height_shift_range=0.1,
                brightness_range=(0.9, 1.1)
            )
            
            # Create infinite generators using while True loop
            def infinite_train_generator():
                while True:
                    for batch in train_datagen.flow(X_train, y_train, batch_size=batch_size):
                        yield batch
            
            def infinite_val_generator():
                while True:
                    for batch in ImageDataGenerator().flow(X_val, y_val, batch_size=batch_size):
                        yield batch
            
            # Calculate steps per epoch
            train_steps = max(1, len(X_train) // batch_size)
            val_steps = max(1, len(X_val) // batch_size)

            # Callbacks
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5, 
                min_lr=1e-7,
                verbose=1
            )
            
            early_stopping = EarlyStopping(
                monitor='val_accuracy', 
                patience=15, 
                restore_best_weights=True,
                verbose=1
            )
            
            fold_model_path = f'pokemon_model_fold_{fold+1}.h5'
            model_checkpoint = ModelCheckpoint(
                fold_model_path, 
                save_best_only=True, 
                monitor='val_accuracy'
            )
            
            csv_logger = CSVLogger(f'training_log_fold_{fold+1}.csv')
            
            # ========= STAGE 1: Frozen Base =========
            history1 = self.model.fit(
                infinite_train_generator(),
                steps_per_epoch=train_steps,
                epochs=epochs // 2,
                validation_data=infinite_val_generator(),
                validation_steps=val_steps,
                callbacks=[reduce_lr, early_stopping, model_checkpoint, csv_logger]
            )
            
            # ========= STAGE 2: Fine-Tuning =========
            print("\nStarting Fine-Tuning Stage")
            self.model = self.fine_tune_model(self.model)
            
            # Create generator with smaller batch size for fine-tuning
            def infinite_fine_tune_generator():
                while True:
                    for batch in train_datagen.flow(X_train, y_train, batch_size=batch_size//2):
                        yield batch
            
            fine_tune_steps = max(1, len(X_train) // (batch_size // 2))
            
            history2 = self.model.fit(
                infinite_fine_tune_generator(),
                steps_per_epoch=fine_tune_steps,
                epochs=epochs // 2,
                validation_data=infinite_val_generator(),
                validation_steps=val_steps,
                callbacks=[reduce_lr, early_stopping, model_checkpoint, csv_logger]
            )
            
            # Combine histories
            combined_history = {k: history1.history[k] + history2.history[k] 
                            for k in history1.history}
            fold_histories.append(combined_history)
                
            # Load the best model for this fold
            self.model = load_model(fold_model_path)
            
            # Evaluate the model
            val_results = self.model.evaluate(X_val, y_val, verbose=1)
            fold_scores.append(val_results[1])  # Accuracy
            
            print(f"\nFold {fold+1} - Validation Accuracy: {val_results[1]:.4f}")
            print(f"Fold {fold+1} - Top-3 Accuracy: {val_results[2]:.4f}")
            
            # Save best model across folds
            if val_results[1] > best_score:
                best_score = val_results[1]
                self.model.save(best_model_path)
                print(f"New best model saved with validation accuracy: {best_score:.4f}")
        
        # Load the best model from all folds
        self.model = load_model(best_model_path)
        
        # Plot training history
        self._plot_combined_histories(fold_histories)
        
        # Print average scores
        mean_accuracy = np.mean(fold_scores)
        std_accuracy = np.std(fold_scores)
        print(f"\nMean Validation Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        return fold_histories
    
    def _plot_combined_histories(self, histories):
        """Plot the training history across all folds"""
        plt.figure(figsize=(14, 10))
        
        # Plot accuracy
        plt.subplot(2, 2, 1)
        for i, history in enumerate(histories):
            plt.plot(history['accuracy'], label=f'Fold {i+1}')
        plt.title('Training Accuracy Across Folds')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot validation accuracy
        plt.subplot(2, 2, 2)
        for i, history in enumerate(histories):
            plt.plot(history['val_accuracy'], label=f'Fold {i+1}')
        plt.title('Validation Accuracy Across Folds')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(2, 2, 3)
        for i, history in enumerate(histories):
            plt.plot(history['loss'], label=f'Fold {i+1}')
        plt.title('Training Loss Across Folds')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot validation loss
        plt.subplot(2, 2, 4)
        for i, history in enumerate(histories):
            plt.plot(history['val_loss'], label=f'Fold {i+1}')
        plt.title('Validation Loss Across Folds')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_metrics_all_folds.png')
        plt.close()

    def predict(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img_array = img / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = self.model.predict(img_array)
        predicted_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_idx]
        
        idx_to_label = {v: k for k, v in self.label_mapping.items()}
        predicted_pokemon = idx_to_label[predicted_idx]
        
        # Return top 5 predictions
        top_indices = np.argsort(prediction[0])[-5:][::-1]
        top_predictions = [(idx_to_label[idx], prediction[0][idx]) for idx in top_indices]
        
        return predicted_pokemon, confidence, top_predictions

    def save_model_metadata(self):
        metadata = {
            'label_mapping': self.label_mapping,
            'num_classes': len(self.label_mapping)
        }
        
        with open('pokemon_model_metadata.json', 'w') as f:
            json.dump(metadata, f)

    def verify_image_dir(self):
        """Verify that the image directory exists and contains Pokemon images"""
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
        
        # Get a list of image files
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        
        if len(image_files) == 0:
            raise ValueError(f"No PNG images found in {self.image_dir}")
        
        print(f"Found {len(image_files)} PNG images in {self.image_dir}")
        
        # Print some sample image names to help with debugging
        if len(image_files) > 0:
            print(f"Sample image names: {image_files[:5]}")
            
        return True

    def run_full_pipeline(self, augmentations=15, k_folds=5):
        print("Verifying image directory...")
        self.verify_image_dir()
        
        print("Starting data preprocessing with augmentation...")
        images, labels, numeric_labels = self.preprocess_data(augment=True, num_augmentations=augmentations)
        
        # Check if we have enough data to proceed
        if len(images) == 0:
            raise ValueError("No images were processed. Check your image directory and CSV file.")
        
        print(f"\nCreating and training model for {labels.shape[1]} Pokemon classes...")
        print(f"Using {k_folds}-fold cross validation for better generalization")
        
        histories = self.train_model(images, labels, numeric_labels, k_folds=k_folds)
        
        print("\nSaving model metadata...")
        self.save_model_metadata()
        
        return histories

if __name__ == "__main__":
    # Install required packages if they're not already installed
    try:
        import albumentations
    except ImportError:
        print("Installing albumentations package...")
        import subprocess
        subprocess.check_call(["pip", "install", "albumentations"])
        import albumentations
    
    # Create the classifier with appropriate paths
    csv_path = 'pokemon.csv'
    image_dir = './images'
    
    # Check if paths exist and print helpful diagnostics
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print("Current working directory:", os.getcwd())
        print("Files in current directory:", os.listdir("."))
        print("Please make sure the pokemon.csv file is in the correct location")
        exit(1)
    else:
        print(f"CSV file found at {csv_path}")
        # Print first few lines of CSV to verify contents
        try:
            with open(csv_path, 'r') as f:
                print("First 5 lines of CSV file:")
                for i, line in enumerate(f):
                    if i < 5:
                        print(line.strip())
                    else:
                        break
        except Exception as e:
            print(f"Warning: Could not read CSV file: {e}")
    
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found at {image_dir}")
        print("Current working directory:", os.getcwd())
        print("Please create the images directory and add Pokemon PNG images")
        exit(1)
    else:
        print(f"Image directory found at {image_dir}")
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        print(f"Found {len(image_files)} PNG files in images directory")
        if len(image_files) > 0:
            print("First few images:", image_files[:5])
        else:
            print("Warning: No PNG files found in the images directory!")
    
    print("\n=== Starting Pokemon Classifier ===\n")
    
    classifier = PokemonClassifier(
        csv_path=csv_path,
        image_dir=image_dir
    )
    
    try:
        # Create augmented versions of each Pokemon and use cross-validation
        # Use fewer augmentations and folds while testing to speed things up
        classifier.run_full_pipeline(augmentations=5, k_folds=3)
        
        # Test prediction
        test_image = './images/pikachu.png'
        if os.path.exists(test_image):
            predicted_pokemon, confidence, top5 = classifier.predict(test_image)
            print(f"\nPredicted Pokemon: {predicted_pokemon}")
            print(f"Confidence: {confidence:.2%}")
            print("\nTop 5 predictions:")
            for pokemon, conf in top5:
                print(f"- {pokemon}: {conf:.2%}")
        else:
            print(f"\nTest image not found: {test_image}")
            print("Try another Pokemon image for prediction")
    except Exception as e:
        import traceback
        print(f"Error occurred: {e}")
        traceback.print_exc()
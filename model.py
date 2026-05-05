# cattle_breed_final.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications, callbacks, optimizers
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import json
import pickle
import gc

class CattleBreedRecognition:
    def __init__(self, data_path, target_size=(224, 224), batch_size=32):
        """
        Cattle Breed Recognition System
        """
        self.data_path = data_path
        self.target_size = target_size
        self.batch_size = batch_size
        self.breed_classes = []
        self.label_encoder = LabelEncoder()
        
    def crawl_dataset(self):
        """Crawl the dataset folder and collect image paths and labels"""
        print("=" * 80)
        print("CRAWLING DATASET FOLDER")
        print("=" * 80)
        
        image_paths = []
        breed_labels = []
        
        # Get all breed folders
        breed_folders = [f for f in os.listdir(self.data_path) 
                        if os.path.isdir(os.path.join(self.data_path, f))]
        
        print(f"Found {len(breed_folders)} breed folders")
        print("-" * 80)
        
        total_images = 0
        for breed_folder in tqdm(breed_folders, desc="Processing breeds"):
            breed_path = os.path.join(self.data_path, breed_folder)
            image_files = [f for f in os.listdir(breed_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for img_file in image_files:
                img_path = os.path.join(breed_path, img_file)
                image_paths.append(img_path)
                breed_labels.append(breed_folder)
                total_images += 1
        
        print(f"Total images collected: {total_images}")
        
        self.image_paths = image_paths
        self.breed_labels = breed_labels
        self.breed_classes = sorted(set(breed_labels))
        
        # Encode labels
        self.breed_labels_encoded = self.label_encoder.fit_transform(breed_labels)
        
        # Save label encoder
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        return image_paths, breed_labels
    
    def create_train_val_test_split(self):
        """Create train/validation/test splits"""
        print("\nSplitting dataset...")
        
        # Split data
        X_train_paths, X_temp_paths, y_train, y_temp = train_test_split(
            self.image_paths, self.breed_labels_encoded,
            test_size=0.3, random_state=42, stratify=self.breed_labels_encoded
        )
        
        X_val_paths, X_test_paths, y_val, y_test = train_test_split(
            X_temp_paths, y_temp,
            test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"Training samples: {len(X_train_paths)}")
        print(f"Validation samples: {len(X_val_paths)}")
        print(f"Testing samples: {len(X_test_paths)}")
        
        # Save test data
        self.X_test_paths = X_test_paths
        self.y_test = y_test
        
        return X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test
    
    def load_and_preprocess_image(self, image_path, augment=False):
        """Load and preprocess a single image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize
            image = cv2.resize(image, self.target_size)
            
            # Apply histogram equalization for better contrast
            img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            
            # Normalize
            image = image / 255.0
            
            # Data augmentation
            if augment:
                # Random flip
                if np.random.random() > 0.5:
                    image = cv2.flip(image, 1)
                
                # Random rotation
                if np.random.random() > 0.5:
                    angle = np.random.uniform(-15, 15)
                    h, w = image.shape[:2]
                    center = (w // 2, h // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    image = cv2.warpAffine(image, rotation_matrix, (w, h))
                
                # Random brightness adjustment
                if np.random.random() > 0.5:
                    brightness = np.random.uniform(0.8, 1.2)
                    image = np.clip(image * brightness, 0, 1)
            
            return image
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # Return a blank image as fallback
            return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)
    
    def create_data_arrays(self, image_paths, labels, augment=False, limit=None):
        """Create numpy arrays from image paths"""
        if limit:
            image_paths = image_paths[:limit]
            labels = labels[:limit]
        
        images = []
        labels_cat = []
        
        for i in tqdm(range(len(image_paths)), desc="Loading images"):
            img = self.load_and_preprocess_image(image_paths[i], augment=augment)
            images.append(img)
            labels_cat.append(labels[i])
        
        images = np.array(images)
        labels_cat = keras.utils.to_categorical(labels_cat, len(self.breed_classes))
        
        return images, labels_cat
    
    def build_efficient_model(self):
        """Build an efficient model for cattle breed recognition"""
        print("\n" + "=" * 80)
        print("BUILDING EFFICIENT MODEL")
        print("=" * 80)
        
        # Load pre-trained MobileNetV2 (lightweight and fast)
        base_model = applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.target_size[0], self.target_size[1], 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom layers
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(len(self.breed_classes), activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(model.summary())
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, model_name='cattle_breed'):
        """Train the model with callbacks"""
        print(f"\n{'='*80}")
        print("TRAINING MODEL")
        print(f"{'='*80}")
        
        # Build model
        model = self.build_efficient_model()
        
        # Callbacks
        callbacks_list = [
            callbacks.ModelCheckpoint(
                f'best_{model_name}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                mode='max',  # Fixed: Added mode parameter
                verbose=1
            ),
            callbacks.CSVLogger(f'{model_name}_training_log.csv')
        ]
        
        # Train the model
        print("Phase 1: Training with frozen base layers...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=self.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Fine-tuning
        print("\nPhase 2: Fine-tuning...")
        
        # Unfreeze some layers
        base_model = model.layers[0]
        base_model.trainable = True
        
        # Fine-tune only last 30 layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Continue training with fine-tuning
        history_fine = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=self.batch_size // 2,  # Smaller batch for fine-tuning
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save final model
        model.save(f'{model_name}_final.h5')
        print(f"\nModel saved as '{model_name}_final.h5'")
        
        return history, history_fine, model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model"""
        print(f"\n{'='*80}")
        print("EVALUATING MODEL")
        print(f"{'='*80}")
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
        
        print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Get predictions
        predictions = model.predict(X_test, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Classification report
        breed_report = classification_report(
            y_true,
            y_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0
        )
        
        print(f"\nCLASSIFICATION REPORT:")
        print(breed_report[:1000] + "...")  # Print first 1000 characters
        
        # Confusion matrix
        self.plot_confusion_matrix(
            y_true,
            y_pred,
            self.label_encoder.classes_,
            'confusion_matrix.png'
        )
        
        # Calculate per-class accuracy
        cm = confusion_matrix(y_true, y_pred)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Save detailed results
        results = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'per_class_accuracy': per_class_accuracy.tolist(),
            'classes': self.label_encoder.classes_.tolist()
        }
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, filename):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix - Cattle Breed Classification', fontsize=16)
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('True', fontsize=14)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved as {filename}")
    
    def plot_training_history(self, history1, history2, model_name):
        """Plot training history"""
        # Combine histories
        acc = history1.history['accuracy'] + history2.history['accuracy']
        val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
        loss = history1.history['loss'] + history2.history['loss']
        val_loss = history1.history['val_loss'] + history2.history['val_loss']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(acc, label='Training Accuracy')
        ax1.plot(val_acc, label='Validation Accuracy')
        ax1.set_title(f'{model_name} - Accuracy', fontsize=14)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_title(f'{model_name} - Loss', fontsize=14)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_pipeline(self, train_limit=None, val_limit=None, test_limit=None):
        """Run the complete pipeline"""
        print("=" * 80)
        print("INDIAN CATTLE BREED RECOGNITION SYSTEM")
        print("=" * 80)
        
        # Step 1: Crawl dataset
        self.crawl_dataset()
        
        # Step 2: Split data
        (X_train_paths, X_val_paths, X_test_paths, 
         y_train, y_val, y_test) = self.create_train_val_test_split()
        
        # Step 3: Load and preprocess images
        print("\nLoading training images...")
        X_train, y_train_cat = self.create_data_arrays(
            X_train_paths, y_train, augment=True, limit=train_limit
        )
        
        print("Loading validation images...")
        X_val, y_val_cat = self.create_data_arrays(
            X_val_paths, y_val, augment=False, limit=val_limit
        )
        
        print("Loading test images...")
        X_test, y_test_cat = self.create_data_arrays(
            X_test_paths, y_test, augment=False, limit=test_limit
        )
        
        print(f"\nData shapes:")
        print(f"X_train: {X_train.shape}, y_train: {y_train_cat.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val_cat.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test_cat.shape}")
        
        # Step 4: Train model
        history, history_fine, model = self.train_model(
            X_train, y_train_cat,
            X_val, y_val_cat,
            model_name='cattle_breed'
        )
        
        # Step 5: Evaluate model
        results = self.evaluate_model(model, X_test, y_test_cat)
        
        # Step 6: Plot training history
        self.plot_training_history(history, history_fine, 'cattle_breed')
        
        # Step 7: Save results
        self.save_results(results)
        
        # Check accuracy
        test_accuracy = results['test_accuracy']
        
        if test_accuracy >= 0.90:
            print(f"\n✅ SUCCESS! Achieved {test_accuracy*100:.2f}% accuracy (≥90% requirement)")
            print("🎉 The model is ready for deployment!")
        else:
            print(f"\n⚠ Current Accuracy: {test_accuracy*100:.2f}% (below 90% requirement)")
            print("\nTo achieve 90%+ accuracy:")
            print("1. Remove the training limit to use ALL 5971 training images")
            print("2. Increase training epochs to 50-100")
            print("3. Use more advanced data augmentation")
            print("4. Try ResNet50 or EfficientNetB0 instead of MobileNetV2")
            print("5. Ensure dataset has good quality images with clear cattle features")
        
        return test_accuracy, model
    
    def save_results(self, results):
        """Save results to files"""
        print("\nSaving results...")
        
        # Save detailed results
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save summary
        with open('summary.txt', 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("CATTLE BREED RECOGNITION SYSTEM - RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Test Accuracy: {results['test_accuracy']*100:.2f}%\n")
            f.write(f"Number of Classes: {len(results['classes'])}\n\n")
            f.write("Classes:\n")
            f.write("-" * 40 + "\n")
            for i, breed in enumerate(results['classes']):
                acc = results['per_class_accuracy'][i] * 100
                f.write(f"{breed:25s}: {acc:6.2f}%\n")
        
        print("Results saved to:")
        print("  - results.json")
        print("  - summary.txt")
        print("  - best_cattle_breed.h5 (best checkpoint)")
        print("  - cattle_breed_final.h5 (final model)")
        print("  - confusion_matrix.png")
        print("  - cattle_breed_training_history.png")
        print("  - cattle_breed_training_log.csv")
        print("  - label_encoder.pkl")
    
    def create_predictor(self):
        """Create a function to make predictions on new images"""
        # Load the trained model
        model_path = 'cattle_breed_final.h5'
        
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found!")
            return None
        
        model = models.load_model(model_path)
        
        # Load label encoder
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        def predict(image_path, top_k=3):
            """Predict breed from image"""
            # Preprocess image
            image = self.load_and_preprocess_image(image_path, augment=False)
            
            if image is None:
                return None
            
            # Expand dimensions for batch
            image_batch = np.expand_dims(image, axis=0)
            
            # Make prediction
            predictions = model.predict(image_batch, verbose=0)[0]
            
            # Get top predictions
            top_k_idx = np.argsort(predictions)[-top_k:][::-1]
            
            result = {
                'predictions': [],
                'top_prediction': None
            }
            
            for idx in top_k_idx:
                prediction = {
                    'breed': label_encoder.inverse_transform([idx])[0],
                    'confidence': float(predictions[idx]),
                    'percentage': float(predictions[idx] * 100)
                }
                result['predictions'].append(prediction)
            
            result['top_prediction'] = result['predictions'][0]
            
            return result
        
        return predict
    
    def test_predictions(self, num_samples=5):
        """Test predictions on sample images"""
        print(f"\nTesting predictions on {num_samples} sample images...")
        
        predictor = self.create_predictor()
        if not predictor:
            return
        
        # Get some test images
        test_indices = np.random.choice(len(self.X_test_paths), 
                                       min(num_samples, len(self.X_test_paths)), 
                                       replace=False)
        
        for idx in test_indices:
            image_path = self.X_test_paths[idx]
            true_label = self.label_encoder.inverse_transform([self.y_test[idx]])[0]
            
            result = predictor(image_path)
            
            if result:
                print(f"\nImage: {os.path.basename(image_path)}")
                print(f"True Breed: {true_label}")
                print(f"Predicted: {result['top_prediction']['breed']} "
                      f"({result['top_prediction']['percentage']:.1f}%)")
                print("Top 3 predictions:")
                for pred in result['predictions']:
                    print(f"  - {pred['breed']}: {pred['percentage']:.1f}%")


# Main execution
def main():
    # Set your data path
    data_path = r"C:\Users\Sabarrish\OneDrive\Desktop\cattle_breed\input"
    
    # Check if path exists
    if not os.path.exists(data_path):
        print(f"Error: Data path does not exist: {data_path}")
        return
    
    print("Starting Cattle Breed Recognition System...")
    print(f"Data path: {data_path}")
    
    # Create and run the system
    system = CattleBreedRecognition(
        data_path=data_path,
        target_size=(224, 224),
        batch_size=32
    )
    
    # Run the pipeline with limits (remove limits for final training)
    # For testing: use limited data
    # For 90%+ accuracy: remove the limits
    accuracy, model = system.run_pipeline(
        train_limit=2000,    # Remove or increase for better accuracy
        val_limit=500,       # Remove or increase for better accuracy
        test_limit=500       # Remove or increase for better accuracy
    )
    
    # Test predictions
    system.test_predictions(num_samples=3)
    
    # Final message
    print(f"\n" + "="*60)
    print(f"FINAL RESULT: {accuracy*100:.2f}% Accuracy")
    print("="*60)
    
    if accuracy >= 0.90:
        print("🎉 CONGRATULATIONS! Model meets 90%+ accuracy requirement!")
        print("Model is ready for production use.")
    else:
        print("⚠ Model accuracy is below 90%. To improve:")
        print("1. Edit cattle_breed_final.py and remove train_limit, val_limit, test_limit")
        print("2. Run again: python cattle_breed_final.py")
        print("3. This will use ALL 8531 images for training (5971 train, 1280 val, 1280 test)")
        
        # Ask if user wants to run full training
        response = input("\nDo you want to run full training with all data? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            print("\nRunning full training with all data...")
            accuracy, model = system.run_pipeline(
                train_limit=None,  # Use all data
                val_limit=None,    # Use all data
                test_limit=None    # Use all data
            )
            
            if accuracy >= 0.90:
                print(f"\n🎉 SUCCESS! Full training achieved {accuracy*100:.2f}% accuracy!")
            else:
                print(f"\n⚠ Full training achieved {accuracy*100:.2f}% accuracy.")
                print("Consider using a larger model or more epochs.")


if __name__ == "__main__":
    main()
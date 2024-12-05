# predict_height.py
# made by Sambhav and Shreyanshi 
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import logging

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# ------------------------------
# 1. Data Preparation
# ------------------------------

def load_and_process_image(filename, image_dir, target_size=(224, 224)):
    """
    Loads an image, converts it to RGB, resizes it, and converts to a NumPy array.

    Parameters:
    - filename: Name of the image file.
    - image_dir: Directory where images are stored.
    - target_size: Desired image size.

    Returns:
    - Numpy array of the processed image or None if loading fails.
    """
    path = os.path.join(image_dir, filename)
    if not filename:
        logging.warning("Missing filename. Skipping entry.")
        return None
    if not os.path.exists(path):
        logging.warning(f"Image file '{filename}' does not exist. Skipping entry.")
        return None
    try:
        with Image.open(path) as img:
            img = img.convert('RGB')
            img = img.resize(target_size)
            return np.array(img)
    except Exception as e:
        logging.error(f"Error loading {filename}: {e}")
        return None

def prepare_data(csv_file='data.csv', image_dir='image_dataset', target_size=(224, 224)):
    """
    Loads data from CSV, processes images, and returns feature and label arrays.

    Parameters:
    - csv_file: Path to the CSV file.
    - image_dir: Directory where images are stored.
    - target_size: Desired image size.

    Returns:
    - X: Numpy array of images.
    - y: Numpy array of corresponding heights.
    - data_filtered: Filtered DataFrame with valid images.
    """
    # Load the CSV data
    logging.info("Loading CSV data...")
    data = pd.read_csv(csv_file)

    # Check for NaNs in labels
    if data['Height (mm)'].isnull().any():
        logging.warning("NaN values found in 'Height (mm)' labels. These entries will be removed.")
        data = data.dropna(subset=['Height (mm)'])

    # Filter out entries with missing filenames
    data = data.dropna(subset=['filename'])

    # Load and process images
    logging.info("Loading and processing images...")
    data['image'] = data['filename'].apply(lambda x: load_and_process_image(x, image_dir, target_size))

    # Drop rows where image loading failed
    data = data.dropna(subset=['image'])

    # Extract features and labels
    X = np.stack(data['image'].values)
    y = data['Height (mm)'].values

    logging.info(f"Total samples after loading and preprocessing: {X.shape[0]}")

    return X, y, data

# ------------------------------
# 2. Data Augmentation and Saving
# ------------------------------

def augment_and_save_images(X, y, data, augmentation_factor=40, target_size=(224, 224),
                           augmented_dir='augmented_images', new_csv='augmented_data.csv'):
    """
    Augments the dataset by generating augmented images, saves them to a folder,
    and creates a new CSV mapping augmented images to their labels.

    Parameters:
    - X: Original images.
    - y: Original labels.
    - data: Original DataFrame containing metadata.
    - augmentation_factor: Number of augmented images to generate per original image.
    - target_size: Desired image size.
    - augmented_dir: Directory to save augmented images.
    - new_csv: Filename for the new augmented CSV.

    Returns:
    - None
    """
    if not os.path.exists(augmented_dir):
        os.makedirs(augmented_dir)
        logging.info(f"Created directory '{augmented_dir}' for augmented images.")

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    augmented_data = []

    logging.info("Starting data augmentation and saving augmented images...")

    for i in range(X.shape[0]):
        img = X[i]
        label = y[i]
        original_filename = data.iloc[i]['filename']
        img_expanded = np.expand_dims(img, axis=0)  # Shape: (1, 224, 224, 3)
        aug_iter = datagen.flow(img_expanded, batch_size=1)
        for j in range(augmentation_factor):
            aug_img = next(aug_iter)[0]
            # Generate a unique filename
            base_name, ext = os.path.splitext(original_filename)
            augmented_filename = f"{base_name}_aug_{j+1:03d}{ext}"
            # Create a new row for the augmented CSV
            new_row = {
                's.no': len(augmented_data) + 1,  # Sequential numbering
                'feed(cm/s)': data.iloc[i]['feed(cm/s)'],
                'feed(mm/min)': data.iloc[i]['feed(mm/min)'],
                'current(A)': data.iloc[i]['current(A)'],
                'TS(mm/min)': data.iloc[i]['TS(mm/min)'],
                'Width (mm)': data.iloc[i]['Width (mm)'],
                'Height (mm)': label,
                'filename': augmented_filename
            }
            augmented_data.append(new_row)
            # Save the augmented image
            save_path = os.path.join(augmented_dir, augmented_filename)
            try:
                Image.fromarray(aug_img.astype('uint8')).save(save_path)
                logging.debug(f"Saved augmented image: {augmented_filename}")
            except Exception as e:
                logging.error(f"Failed to save augmented image {augmented_filename}: {e}")

    # Convert augmented data to DataFrame
    augmented_df = pd.DataFrame(augmented_data)

    # Save the augmented DataFrame to CSV
    augmented_df.to_csv(new_csv, index=False)
    logging.info(f"Augmented data CSV saved as '{new_csv}' with {len(augmented_df)} entries.")

# ------------------------------
# 3. Creating Data Generators
# ------------------------------

def create_data_generators(train_df, test_df, augmented_dir, target_size=(224, 224), batch_size=32):
   
    # Define the ImageDataGenerator for training with augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1./255  # Rescaling handled here
    )

    # Define the ImageDataGenerator for testing without augmentation
    test_datagen = ImageDataGenerator(
        rescale=1./255  # Rescaling handled here
    )

    # Create the training generator
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=augmented_dir,
        x_col='filename',
        y_col='Height (mm)',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='raw',  # 'raw' is used for regression
        shuffle=True
    )

    # Create the testing generator
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=augmented_dir,
        x_col='filename',
        y_col='Height (mm)',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='raw',  # 'raw' is used for regression
        shuffle=False
    )

    return train_generator, test_generator

# ------------------------------
# 4. Building the CNN Model
# ------------------------------

def build_cnn_model(input_shape, learning_rate=1e-4):
    """
    Builds and compiles the CNN model using MobileNetV2 as the base.

    Parameters:
    - input_shape: Shape of the input images.
    - learning_rate: Learning rate for the optimizer.

    Returns:
    - model: Compiled Keras model.
    """
    # Load the pre-trained MobileNetV2 model without the top classification layers
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base model
    base_model.trainable = False

    # Add custom layers on top for regression
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')  # Regression output
    ])

    model.summary()

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model

# ------------------------------
# 5. Training the Model
# ------------------------------

def train_model(model, train_generator, test_generator, epochs=10, checkpoint_path='best_model.keras'):
    """
    Trains the CNN model with the given data generators.

    Parameters:

    - train_generator: Data generator for training.
    - test_generator: Data generator for testing.
    - epochs: Maximum number of training epochs.
    - checkpoint_path: Path to save the best model.

  
    """
    # Define callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Calculate steps per epoch
    steps_per_epoch = int(np.ceil(train_generator.n / train_generator.batch_size))
    validation_steps = int(np.ceil(test_generator.n / test_generator.batch_size))

    logging.info(f"Steps per epoch: {steps_per_epoch}")
    logging.info(f"Validation steps: {validation_steps}")

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=validation_steps,
        callbacks=[early_stop, checkpoint]
    )

    return history

# ------------------------------
# 6. Evaluation and Visualization
# ------------------------------

def evaluate_and_plot(model, test_generator):
    """
    Evaluates the trained model on the test set and plots Actual vs Predicted heights.

    Parameters:
    - model: Trained Keras model.
    - test_generator: Data generator for testing.

    Returns:
    - None
    """
    # Reset the test generator
    test_generator.reset()

    # Make predictions
    y_pred = model.predict(test_generator, steps=int(np.ceil(test_generator.n / test_generator.batch_size)))
    y_pred = y_pred.flatten()

    # Get true labels
    y_true = test_generator.labels[:len(y_pred)]

    # Calculate evaluation metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    logging.info(f"\nTest Mean Squared Error (MSE): {mse:.4f}")
    logging.info(f"Test Mean Absolute Error (MAE): {mae:.4f}")
    logging.info(f"Test R² Score: {r2:.4f}")

    # Plot Actual vs Predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='b')
    plt.xlabel('Actual Height (mm)')
    plt.ylabel('Predicted Height (mm)')
    plt.title('Actual vs. Predicted Heights')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # Diagonal line
    plt.grid(True)
    plt.show()

# ------------------------------
# 7. Optional Fine-Tuning
# ------------------------------

def fine_tune_model(model, base_model, train_generator, test_generator, fine_tune_epochs=100, learning_rate=1e-5, checkpoint_path='best_model_finetuned.keras'):
    """
    Fine-tunes the pre-trained base model to potentially improve performance.

    Parameters:
    - model: Keras model with the base model frozen.
    - base_model: The pre-trained base model.
    - train_generator: Data generator for training.
    - test_generator: Data generator for testing.
    - fine_tune_epochs: Maximum number of fine-tuning epochs.
    - learning_rate: Learning rate for the optimizer.
    - checkpoint_path: Path to save the fine-tuned model.

    Returns:
    - history_ft: Fine-tuning training history object.
    """
   
    base_model.trainable = True

    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mae']
    )

  
    early_stop_ft = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    checkpoint_ft = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

 
    steps_per_epoch = int(np.ceil(train_generator.n / train_generator.batch_size))
    validation_steps = int(np.ceil(test_generator.n / test_generator.batch_size))

    logging.info(f"Fine-Tuning Steps per epoch: {steps_per_epoch}")
    logging.info(f"Fine-Tuning Validation steps: {validation_steps}")

  
    history_ft = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=fine_tune_epochs,
        validation_data=test_generator,
        validation_steps=validation_steps,
        callbacks=[early_stop_ft, checkpoint_ft]
    )

    return history_ft

# ------------------------------
# Main Execution
# ------------------------------

def main():
    # Parameters
    csv_file = 'data.csv'
    image_dir = 'image_dataset'
    augmented_dir = 'augmented_images'
    new_csv = 'augmented_data.csv'
    target_size = (224, 224)
    batch_size = 32
    initial_learning_rate = 1e-4
    fine_tune_learning_rate = 1e-5
    epochs = 10
    fine_tune_epochs = 10
    checkpoint_path = 'best_model.keras'
    fine_tune_checkpoint_path = 'best_model_finetuned.keras'

    # Step 1: Data Preparation
    logging.info("Step 1: Data Preparation")
    X_original, y_original, data_filtered = prepare_data(csv_file, image_dir, target_size)

    # Step 2: Data Augmentation and Saving
    logging.info("\nStep 2: Data Augmentation and Saving")
    augment_and_save_images(X_original, y_original, data_filtered, augmentation_factor=40, target_size=target_size,
                           augmented_dir=augmented_dir, new_csv=new_csv)

    # Step 3: Load and Verify Augmented Data
    logging.info("\nStep 3: Load and Verify Augmented Data")
    augmented_df = pd.read_csv(new_csv)
    # Verify that all augmented images exist
    augmented_df_verified = augmented_df[augmented_df['filename'].apply(lambda x: os.path.isfile(os.path.join(augmented_dir, x)))]
    missing_augmented = augmented_df[~augmented_df['filename'].apply(lambda x: os.path.isfile(os.path.join(augmented_dir, x)))]
    if not missing_augmented.empty:
        logging.warning(f"Found {missing_augmented.shape[0]} missing augmented images. These will be excluded from the dataset.")
        logging.warning(missing_augmented[['s.no', 'filename']])
        # Exclude missing augmented images
        augmented_df_verified = augmented_df_verified
    else:
        logging.info("All augmented images are present.")

    # Step 4: Split into Train and Test (80-20)
    logging.info("\nStep 4: Splitting into Training and Testing Sets (80-20)")
    train_df, test_df = train_test_split(augmented_df_verified, test_size=0.2, random_state=42)
    logging.info(f"Training samples: {train_df.shape[0]}")
    logging.info(f"Testing samples: {test_df.shape[0]}")

    # Step 5: Create Data Generators
    logging.info("\nStep 5: Creating Data Generators")
    train_generator, test_generator = create_data_generators(train_df, test_df, augmented_dir, target_size, batch_size)
    logging.info("Data generators created.")

    # Step 6: Build the CNN Model
    logging.info("\nStep 6: Building the CNN Model")
    input_shape = target_size + (3,)  # (224, 224, 3)
    model = build_cnn_model(input_shape, learning_rate=initial_learning_rate)

    # Step 7: Train the Model
    logging.info("\nStep 7: Training the Model")
    history = train_model(model, train_generator, test_generator, epochs, checkpoint_path)

    # Step 8: Evaluate the Model
    logging.info("\nStep 8: Evaluating the Model on the Test Set")
    model.load_weights(checkpoint_path)
    evaluate_and_plot(model, test_generator)

    # Step 9: Optional Fine-Tuning
    logging.info("\nStep 9: Optional Fine-Tuning")
    fine_tune = input("Do you want to perform fine-tuning? (yes/no): ").strip().lower()
    if fine_tune == 'yes':
        logging.info("Starting fine-tuning...")
        base_model = model.layers[0]  # Assuming the base model is the first layer
        history_ft = fine_tune_model(model, base_model, train_generator, test_generator,
                                     fine_tune_epochs, fine_tune_learning_rate, fine_tune_checkpoint_path)
        logging.info("Fine-tuning completed.")

        # Evaluate the fine-tuned model
        logging.info("\nEvaluating the Fine-Tuned Model on the Test Set")
        model.load_weights(fine_tune_checkpoint_path)
        evaluate_and_plot(model, test_generator)
    else:
        logging.info("Fine-tuning skipped.")

    logging.info("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()

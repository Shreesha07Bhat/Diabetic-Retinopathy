# Improved train_model.py with Transfer Learning using ResNet50

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Path to the dataset
dataset_path = './'
images_dir = os.path.join(dataset_path, 'gaussian_filtered_images', 'gaussian_filtered_images')
csv_file = os.path.join(dataset_path, 'train.csv')

# Load the CSV file
df = pd.read_csv(csv_file)
print("CSV Columns:", df.columns)

# Convert 'diagnosis' to integer
df['diagnosis'] = df['diagnosis'].astype(int)

# Define the mapping from numerical diagnosis to descriptive labels
diagnosis_mapping = {
    '0': 'No_DR',
    '1': 'Mild',
    '2': 'Moderate',
    '3': 'Severe',
    '4': 'Proliferate_DR'
}

# Map numerical labels to descriptive labels
df['diagnosis'] = df['diagnosis'].astype(str).map(diagnosis_mapping)

# Function to get image path with diagnosis subdirectory
def get_image_path(row, images_dir):
    id_code = row['id_code']
    diagnosis = row['diagnosis']
    for ext in ['.jpg', '.jpeg', '.png']:
        path = os.path.join(images_dir, diagnosis, f"{id_code}{ext}")
        if os.path.exists(path):
            return path
    print(f"Image not found for id_code: {id_code} in diagnosis: {diagnosis}")
    return None

# Create file paths
df['filepath'] = df.apply(lambda row: get_image_path(row, images_dir), axis=1)

# Drop rows with missing filepaths
initial_count = len(df)
df = df.dropna(subset=['filepath'])
final_count = len(df)
print(f"Dropped {initial_count - final_count} rows due to missing filepaths.")

# Verify filepaths
print("\nSample Filepaths:")
print(df['filepath'].head())

print("\nFilepath Existence Check:")
print(df['filepath'].apply(os.path.exists).value_counts())

# Check if there are valid filepaths before splitting
if final_count == 0:
    raise ValueError("No valid image filepaths found. Please check your dataset and file paths.")

# Split the dataframe into training and validation sets with stratification
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['diagnosis']
)
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# Data generators with augmentation for training and rescaling for validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filepath',
    y_col='diagnosis',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_dataframe(
    val_df,
    x_col='filepath',
    y_col='diagnosis',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load the ResNet50 model without the top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(diagnosis_mapping), activation='softmax')(x)

# Define the complete model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[early_stop, checkpoint]
)

# Save the final model
model.save('model_final.h5')
print("Model training complete and saved as 'model_final.h5'")
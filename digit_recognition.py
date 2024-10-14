import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.datasets import mnist  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data: reshape and normalize
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255  # cSpell error: "astype"
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255  # cSpell error: "astype"

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

# Data augmentation
data_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1)  # cSpell error: "datagen"
data_gen.fit(X_train)  # cSpell error: "datagen"

# Build a more robust CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Corrected "relU" to "relu"
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),  # Corrected "relU" to "relu"
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),  # Corrected "relU" to "relu"
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),  # Corrected "relU" to "relu"
    layers.Dropout(0.5),  # Added dropout for regularization
    layers.Dense(10, activation='softmax')  # Corrected "softMax" to "softmax"
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # cSpell error: "crossentropy"
              metrics=['accuracy'])

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with data augmentation
history = model.fit(data_gen.flow(X_train, y_train, batch_size=64),  # cSpell error: "datagen"
                    epochs=20,  # Increased epochs for better training
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping])

# Save the model after training
model.save('handwritten_digit_model.h5')

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Print model summary
model.summary()

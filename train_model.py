import json
import os

import tensorflow as tf
from tensorflow.keras import layers, models


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_image_classifier.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42


def build_cnn_model(num_classes):
    model = models.Sequential(
        [
            layers.Input(shape=(128, 128, 3)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4), #Randomly removes 40% neurons.
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError("Dataset folder was not found.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    print("Class names:", class_names)

    normalization_layer = layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
       #Stores dataset in memory, Randomizes data, Loads next batch while current batch is processing
    train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    model = build_cnn_model(num_classes=len(class_names))
    model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)

    loss, accuracy = model.evaluate(test_ds)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True) #Create Model Folder
    model.save(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as class_file:
        json.dump(class_names, class_file)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Class names saved to: {CLASS_NAMES_PATH}")


if __name__ == "__main__":
    main()

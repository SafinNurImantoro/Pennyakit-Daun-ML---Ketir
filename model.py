import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback, EarlyStopping
import os

def build_model(input_shape=(224, 224, 3), num_classes=3):
    base_model = MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(dataset_path, epochs=15):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    model = build_model(num_classes=len(train_gen.class_indices))

    class PrintProgress(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"Epoch {epoch+1}/{epochs} - Acc: {logs['accuracy']:.2f}, Val_Acc: {logs['val_accuracy']:.2f}")

    callbacks = [
        PrintProgress(),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=4
    )

    os.makedirs("saved_model", exist_ok=True)
    model.save("saved_model/leaf_model.h5")
    with open("saved_model/label_map.txt", "w") as f:
        for label, idx in train_gen.class_indices.items():
            f.write(f"{label},{idx}\n")

    return model, train_gen.class_indices, history.history

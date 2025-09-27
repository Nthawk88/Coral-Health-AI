import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix, classification_report

train_dir = 'coral_dataset'

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print('Class indices:', train_generator.class_indices)
num_classes = len(train_generator.class_indices)

labels = sorted(train_generator.class_indices.items(), key=lambda kv: kv[1])
ordered_class_names = [name for name, idx in labels]
os.makedirs('model', exist_ok=True)
with open('model/labels.json', 'w', encoding='utf-8') as f:
    json.dump({'classes': ordered_class_names}, f, ensure_ascii=False, indent=2)

base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_path = 'model/coral_model.h5'
os.makedirs('model', exist_ok=True)
cb = [
    callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
]

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=40,
    callbacks=cb
)

model.save(checkpoint_path)

val_steps = max(1, validation_generator.samples // validation_generator.batch_size)
validation_generator.reset()
preds = model.predict(validation_generator, steps=val_steps, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = validation_generator.classes[:len(y_pred)]

labels_idx = list(train_generator.class_indices.values())
target_names = list(train_generator.class_indices.keys())
print('\nClassification Report:')
print(classification_report(y_true, y_pred, labels=labels_idx[:num_classes], target_names=target_names))
print('\nConfusion Matrix:')
print(confusion_matrix(y_true, y_pred, labels=labels_idx[:num_classes]))

with open('model/training_summary.txt', 'w', encoding='utf-8') as f:
    f.write(str(history.history))

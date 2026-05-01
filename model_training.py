import os
import numpy as np
from data_preprocessor import preprocess_and_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

data_dir = "data_storage"
default_file = "GSPC_clean_data.csv"
print("************************************************")
print("--- Machine Learning Data Loader ---")
file_prompt = input(f"Enter the CLEANED filename to load (default: {default_file}): ").strip()
target_file = file_prompt if file_prompt else default_file
if not target_file.lower().endswith('.csv'):
    target_file += '.csv'

input_path = os.path.join(data_dir, target_file)

print("************************************************")
print(f"MODEL TRAINING")
print("************************************************")

def build_model(input_dim: int):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(16, activation='relu'),
        layers.Dropout(0.1),

        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model

X_train, X_test, y_train, y_test = preprocess_and_split(input_path)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-5
)
model = build_model(X_train.shape[1])

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    shuffle=False,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)

print("\n==========================")
print("TEST RESULTS")
print("==========================")
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")

save_path = "models/stock_model.keras"
os.makedirs("models", exist_ok=True)

model.save(save_path)

print(f"\nModel saved to: {save_path}")
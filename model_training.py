"""
model_training.py
"""

import os
import glob
import data_gen
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot
from random import shuffle

# Configuration parameters
DEVICE = "/CPU:0"  # Change to "/GPU:0" if GPU available

# Automatically find project folder and processed dataset
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# processed_data is inside rover_data/
DATA_PATH = os.path.join(SCRIPT_DIR, "rover_data", "processed_data")

MODEL_NUM = 1
TRAINING_VER = 1

NUM_EPOCHS = 15
BATCH_SIZE = 32
TRAIN_VAL_SPLIT = 0.8

IMG_H, IMG_W = 120, 160

# Not using temporal modeling
SEQUENCE_SIZE = 1


# ---------------- MODEL ----------------

def define_model(input_shape=(120, 160, 1)):
    model = Sequential([

        Conv2D(16, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),

        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(128, activation="relu"),
        Dropout(0.3),

        Dense(2)  # [steering_norm, throttle_norm]
    ])

    return model


# ---------------- TRAINING ----------------

def train_model(amt_data=1.0):

    # ✅ ONLY CHANGE: load samples from PNGs directly inside each run folder
    allowed_runs = [
        "cloning_20260304-175703",
        "cloning_20260304-181035",
        "cloning_20260304-181318"
    ]

    samples = []
    for run in allowed_runs:
        run_path = os.path.join(DATA_PATH, run)

        pattern = os.path.join(run_path, "*_bw.png")
        run_files = sorted(glob.glob(pattern))

        samples.extend(run_files)

    print(f"Total samples found: {len(samples)}")

    shuffle(samples)

    if amt_data < 1.0:
        samples, _ = data_gen.split_samples(samples, fraction=amt_data)

    train_samples, val_samples = data_gen.split_samples(
        samples,
        fraction=TRAIN_VAL_SPLIT
    )

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    train_steps = max(1, int(len(train_samples) / BATCH_SIZE))
    val_steps = max(1, int(len(val_samples) / BATCH_SIZE))

    print(f"Steps per epoch: {train_steps}")
    print(f"Val steps: {val_steps}")

    train_gen = data_gen.batch_generator(
        train_samples,
        batch_size=BATCH_SIZE,
        img_h=IMG_H,
        img_w=IMG_W
    )

    val_gen = data_gen.batch_generator(
        val_samples,
        batch_size=BATCH_SIZE,
        img_h=IMG_H,
        img_w=IMG_W
    )

    os.makedirs("models", exist_ok=True)

    with tf.device(DEVICE):

        model = define_model(input_shape=(IMG_H, IMG_W, 1))

        model.summary()

        filePath = (
            "models/rover_model_"
            + f"{MODEL_NUM:02d}_ver{TRAINING_VER:02d}"
            + "_epoch{epoch:04d}_val_loss{val_loss:.4f}.h5"
        )

        checkpoint_best = ModelCheckpoint(
            filePath,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min"
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        model.compile(
            optimizer=Adam(5e-4),
            loss="mse"
        )

        history = model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=NUM_EPOCHS,
            callbacks=[checkpoint_best, early_stop],
            verbose=1
        )

    return history



def summarize_diagnostics(histories):

    for i in range(len(histories)):

        pyplot.subplot(len(histories), 1, i + 1)

        pyplot.title("Training Loss Curves")

        pyplot.plot(
            histories[i].history["loss"],
            color="blue",
            label="train"
        )

        pyplot.plot(
            histories[i].history["val_loss"],
            color="orange",
            label="validation"
        )

        pyplot.legend()

    pyplot.show()



def main():

    history = train_model(1.0)

    summarize_diagnostics([history])


if __name__ == "__main__":
    main()
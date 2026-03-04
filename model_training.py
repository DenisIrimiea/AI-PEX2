"""
model_training.py

"""

import data_gen
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot

# Configuration parameters
DEVICE = "/CPU:0"  # Device to use for computation. Change to "/GPU:0" if GPU is available
import os
DATA_PATH = os.path.expanduser(
    "~/Downloads/AI-PEX2/Cripple/rover_data_processed"
)

MODEL_NUM = 1 # Model number for naming
TRAINING_VER = 1  # Training version for naming

NUM_EPOCHS = 15  # You can set high because EarlyStopping will stop early
BATCH_SIZE = 32  # More standard than 35/13
TRAIN_VAL_SPLIT = 0.8  # Train/validation split ratio
IMG_H, IMG_W = 120, 160

# ✅ You are not doing temporal modeling, so don't force sequences of 13
SEQUENCE_SIZE = 1

# Define the CNN model structure
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


# Train the model with data from a generator, using checkpoints and a specified device
def train_model(amt_data=1.0):
    
    # Load samples (i.e. preprocessed frames for training).
    # Note that we are using sequences consisting of 13 frames.
    samples = data_gen.get_sequence_samples(DATA_PATH, sequence_size=SEQUENCE_SIZE)

    print(f"Total samples found: {len(samples)}")
    
    # You may wish to do simple testing using only 
    # a fraction of your training data...
    if amt_data < 1.0:
        # Use only a portion of the entire dataset
        samples, _\
            = data_gen.split_samples(samples, fraction=amt_data)

    # Now, split our samples into training and validation sets
    # Note that train_samples will contain a flat list of sequenced 
    # image file paths.
    train_samples, val_samples = data_gen.split_samples(samples, fraction=TRAIN_VAL_SPLIT)

    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    train_steps = max(1, int(len(train_samples) / BATCH_SIZE))
    val_steps = max(1, int(len(val_samples) / BATCH_SIZE))

    print(f"Steps per epoch: {train_steps} | Val steps: {val_steps}")

    # Create data generators that will supply both the training and validation data during training.
    train_gen = data_gen.batch_generator(train_samples, batch_size=BATCH_SIZE, img_h=IMG_H, img_w=IMG_W)
    val_gen = data_gen.batch_generator(val_samples, batch_size=BATCH_SIZE, img_h=IMG_H, img_w=IMG_W)
    
    # ✅ Ensure output dir exists
    os.makedirs("models", exist_ok=True)

    with tf.device(DEVICE):

        # Note that your input shape must match your preprocessed image size
        model = define_model(input_shape=(IMG_H, IMG_W, 1))
        model.summary()  # Print a summary of the model architecture
        
        # Path for saving the best model checkpoints
        filePath = "models/rover_model_" + f"{MODEL_NUM:02d}_ver{TRAINING_VER:02d}" + "_epoch{epoch:04d}_val_loss{val_loss:.4f}.h5"
        
        # Save only the best (i.e. min validation loss) epochs
        checkpoint_best = ModelCheckpoint(filePath, monitor="val_loss", 
                                          verbose=1, save_best_only=True, 
                                          mode="min")
        
        # ✅ Stop when validation stops improving (prevents overfitting on small datasets)
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        # Train your model here.
        model.compile(optimizer=Adam(5e-4), loss="mse")  # slightly smaller LR = steadier training
        history = model.fit(train_gen,
                            steps_per_epoch=train_steps,
                            validation_data=val_gen,
                            validation_steps=val_steps,
                            epochs=NUM_EPOCHS,
                            callbacks=[checkpoint_best, early_stop],
                            verbose=1)

        #print(history.history.keys())
    return history


# Plot training and validation loss over epochs
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(len(histories),1,i+1)
        pyplot.title('Training Loss Curves')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
        pyplot.legend()

    pyplot.show()

# Run the training process and display training diagnostics
def main():
    history = train_model(1.0)  # ✅ use all data
    summarize_diagnostics([history])


# Entry point to start the training process
if __name__ == "__main__":
    main()
"""
model_training.py

"""

import data_gen
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot

# Configuration parameters
DEVICE = "/CPU:0"  # Device to use for computation. Change to "/GPU:0" if GPU is available
DATA_PATH = "/media/data/rover_data/processed/smooth/left"  # Path to the processed data
MODEL_NUM = 1 # Model number for naming
TRAINING_VER = 1  # Training version for naming
NUM_EPOCHS = 5  # Number of epochs to train
BATCH_SIZE = 13  # Batch size for training
TRAIN_VAL_SPLIT = 0.8  # Train/validation split ratio


# Define the CNN model structure
def define_model(input_shape=(80, 160, 1)):
    
    model = Sequential()

    return model

# Train the model with data from a generator, using checkpoints and a specified device
def train_model(amt_data=1.0):
    
    # Load samples (i.e. preprocessed frames for training).
    # Note that we are using sequences consisting of 13 frames.
    samples = data_gen.get_sequence_samples(DATA_PATH, sequence_size=13)
    
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

    train_steps = int(len(train_samples) / BATCH_SIZE)
    val_steps = int(len(val_samples) / BATCH_SIZE)

    # Create data generators that will supply both the training and validation data during training.
    train_gen = data_gen.batch_generator(train_samples, batch_size=BATCH_SIZE)
    val_gen = data_gen.batch_generator(val_samples, batch_size=BATCH_SIZE)
    
    with tf.device(DEVICE):

        # Note that your input shape must match your preprocessed image size
        model = define_model(input_shape=(160, 320, 1))
        model.summary()  # Print a summary of the model architecture
        
        # Path for saving the best model checkpoints
        filePath = "models/rover_model_" + f"{MODEL_NUM:02d}_ver{TRAINING_VER:02d}" + "_epoch{epoch:04d}_val_loss{val_loss:.4f}.h5"
        
        # Save only the best (i.e. min validation loss) epochs
        checkpoint_best = ModelCheckpoint(filePath, monitor="val_loss", 
                                          verbose=1, save_best_only=True, 
                                          mode="min")
        
        # Train your model here.
        history=None
        #history = model.fit(...)

        #print(history.history.keys())
    return history


# Plot training and validation loss over epochs
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(len(histories),1,1)
        pyplot.title('Training Loss Curves')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')

    pyplot.show()

# Run the training process and display training diagnostics
def main():
    history = train_model(0.06)
    summarize_diagnostics([history])


# Entry point to start the training process
if __name__ == "__main__":
    main()

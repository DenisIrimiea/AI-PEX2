"""
data_gen.py
"""

import cv2
import os
import glob
from random import shuffle
import numpy as np

# Constants defining the range of steering and throttle values
STEERING_MIN = 1176
STEERING_MAX = 2006
THROTTLE_MIN = 985
THROTTLE_MAX = 1766

# Normalizes a value to a 0-1 scale based on a minimum and maximum value
def min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val - v_min) / (v_max - v_min)

# Inverts a normalized value back to its original scale
def invert_min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val * (v_max - v_min)) + v_min

# Gathers a list of image file paths in sequences from subdirectories of a root folder
def get_sample_series_list(root_folder, sequence_size=13,
                           offset_start=0, shuffle_series=True,
                           random_state=None, interval=1, ends_with="*_bw.png"):

    samples = []
    sub_folders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    sequence = []

    for folder in sub_folders:
        path = os.path.join(folder, ends_with)
        files = sorted(glob.glob(path))
        file_count = 0
        for file in files:
            if file_count % interval == 0:
                if len(sequence) >= sequence_size:
                    samples.append(sequence)
                    sequence = []

                file_count += 1
                if file_count >= offset_start:
                    sequence.append(file)
            else:
                file_count += 1

        # (optional) flush remainder for this folder
        if len(sequence) >= sequence_size:
            samples.append(sequence)
        sequence = []

    if shuffle_series:
        # ✅ FIX: random.shuffle doesn't accept random=
        shuffle(samples)

    return samples


def get_sequence_samples(root_folder, sequence_size=13,
                         offset_start=0, shuffle_series=True,
                         random_state=None, interval=1):

    samples = get_sample_series_list(root_folder=root_folder,
                                     sequence_size=sequence_size,
                                     offset_start=offset_start,
                                     shuffle_series=shuffle_series,
                                     random_state=random_state,
                                     interval=interval)

    samples = [item for sublist in samples for item in sublist]
    return samples


def split_samples(samples, fraction=0.8):
    length = len(samples)
    num_training = int(fraction * length)
    return samples[:num_training], samples[num_training:]


def batch_generator(samples, batch_size=13,
                    normalize_labels=True,
                    y_min=1000.0, y_max=2000.0,
                    img_h=120, img_w=160):

    num_samples = len(samples)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            labels = []
            for batch_sample in batch_samples:
                try:
                    file_name = os.path.basename(batch_sample).replace(".png", "")

                    attributes = file_name.split('_')
                    if len(attributes) < 5:
                        f_num, throttle, steering, f_type = file_name.split('_')
                    else:
                        f_num, throttle, steering, heading, f_type = file_name.split('_')

                    throttle = int(throttle)
                    steering = int(steering)

                    image = cv2.imread(batch_sample, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        raise FileNotFoundError("cv2.imread returned None")

                    # ✅ Make sure size is consistent
                    if image.shape != (img_h, img_w):
                        image = cv2.resize(image, (img_w, img_h), interpolation=cv2.INTER_AREA)

                    # ✅ Normalize image and add channel dimension for CNN
                    image = image.astype(np.float32) / 255.0
                    image = np.expand_dims(image, axis=-1)  # (H,W,1)
                    images.append(image)

                    if normalize_labels:
                        # ✅ Use correct ranges (separate steering vs throttle)
                        steering_n = min_max_norm(steering, STEERING_MIN, STEERING_MAX)
                        throttle_n = min_max_norm(throttle, THROTTLE_MIN, THROTTLE_MAX)
                    else:
                        steering_n = float(steering)
                        throttle_n = float(throttle)

                    labels.append([steering_n, throttle_n])

                except Exception as e:
                    print(f" [EXCEPTION ENCOUNTERED: {e}; skipping sample {batch_sample}.] ")

            x_train = np.array(images, dtype=np.float32)
            y_train = np.array(labels, dtype=np.float32)

            yield x_train, y_train
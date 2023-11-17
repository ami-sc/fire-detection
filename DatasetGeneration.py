import os
import cv2
import pandas as pd

from WaveletFeatureExtraction import get_wavelet_features

# Change this to switch datasets.
dataset_name = "effective_forest_fire_detection"

fire_dir = f"data/{dataset_name}/fire"
non_fire_dir = f"data/{dataset_name}/nonfire"

wavelet_features = []

print("[ > ] Processing FIRE images.")

# Load fire images.
fire_img_files = os.listdir(fire_dir)
i = 1

for fire_img_file in fire_img_files:
    print(f"      [ > ] Extracting features for FIRE image {i}.")

    # Load the image.
    fire_img = cv2.imread(os.path.join(fire_dir, fire_img_file))

    # Get wavelet features.
    wav = get_wavelet_features(fire_img)

    # Add the label of the image (1 - FIRE).
    wav += [1]

    # Save extracted features.
    wavelet_features.append(wav)
    i += 1

print("[ > ] Finished processing FIRE images.\n")

print("[ > ] Processing NON-FIRE images.")

# Load non-fire images.
non_fire_img_files = os.listdir(non_fire_dir)
i = 1

for non_fire_img_file in non_fire_img_files:
    print(f"      [ > ] Extracting features for NON-FIRE image {i}.")

    # Load the image.
    non_fire_img = cv2.imread(os.path.join(non_fire_dir, non_fire_img_file))

    # Get wavelet features.
    wav = get_wavelet_features(non_fire_img)

    # Add the label of the image (0 - NON-FIRE).
    wav += [0]

    # Save extracted features.
    wavelet_features.append(wav)
    i += 1

print("[ > ] Finished processing NON-FIRE images.\n")

# Create a feature DataFrame.
dataset = pd.DataFrame(data=wavelet_features)

# Randomize the order of the samples.
dataset = dataset.sample(frac=1)
dataset.reset_index(drop=True, inplace=True)

# Save generated features to a .csv file.
dataset.to_csv(f"data/WAV__{dataset_name}.csv", index=False, header=False)

print("[ > ] Saved wavelet feature dataset.")

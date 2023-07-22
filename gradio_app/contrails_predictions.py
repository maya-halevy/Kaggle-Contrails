import os
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import io
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


def fig2img(fig, dpi=300):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi)
    buf.seek(0)
    img = Image.open(buf)
    return img


# Function to load a band file
def load_band_file(path):
    return np.load(path)


# Function to load a mask file
def load_mask_file(path):
    return np.load(path)


# Function to sort all the band file paths from gradio object
def sort_band_paths(file_paths):
    sorted_file_paths = sorted(file_paths, key=lambda path: int(path.split("_")[1].split(".")[0]))
    return sorted_file_paths


def dice_coef_pred(y_true, y_pred):
    y_true_f = K.flatten(K.constant(y_true))
    y_pred_f = K.flatten(K.constant(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef_pred(y_true, y_pred):
    y_true_f = K.flatten(K.constant(y_true))
    y_pred_f = K.flatten(K.constant(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


best_model = load_model('3ch_gen1_-0.546_-0.492_model.h5',
                        compile=False)


def visualize_band_images(band_paths, figsize=(10, 10), cmap='Blues'):
    # Set the figure size
    sorted_band_paths = sort_band_paths(band_paths)

    fig = plt.figure(figsize=figsize)

    # Loop over the bands
    for i in range(9):
        # Load the band data
        band_data = np.load(sorted_band_paths[i])

        # Extract the 5th dimension for visualization
        band_data_5th = band_data[:, :, 4]

        # Create a subplot for each band
        plt.subplot(3, 3, i + 1)

        # Use seaborn to visualize the image
        sns.heatmap(band_data_5th, cmap=cmap, cbar=False)

        # Set the title of the subplot
        plt.title(f'Band {i + 8}')

        plt.axis('off')

    # Adjust the layout and display the plot
    plt.tight_layout()

    # Convert the figure to a PIL Image
    bands_fig_image = fig2img(fig)
    plt.close(fig)

    return bands_fig_image


def visualize_masks(predicted_mask_avg_binary, true_mask):
    combined_colors = np.zeros_like(predicted_mask_avg_binary)
    combined_colors[(predicted_mask_avg_binary == 0) & (true_mask == 0)] = 0
    combined_colors[(predicted_mask_avg_binary == 1) & (true_mask == 1)] = 1
    combined_colors[(predicted_mask_avg_binary == 1) & (true_mask == 0)] = 2
    combined_colors[(predicted_mask_avg_binary == 0) & (true_mask == 1)] = 3

    # Create a custom colormap with four colors
    cmap_colors = ['white', 'green', 'red', '#27c1f5cc']
    cmap = ListedColormap(cmap_colors, N=4)

    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Display the combined_colors array using the custom colormap
    ax.imshow(combined_colors, cmap=cmap)

    legend_elements = [
        mpatches.Patch(facecolor='green', edgecolor='black', label='correct'),
        mpatches.Patch(facecolor='red', edgecolor='black', label='false'),
        mpatches.Patch(facecolor='#27c1f5cc', edgecolor='black', label='unidentified')
    ]

    # Add the legend at the bottom center
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05))

    ax.set_title('True vs Predicted Mask', fontsize=20, y=1.05)

    # Convert the figure to a PIL Image and return it
    combined_mask_img = fig2img(fig)
    plt.close(fig)

    return combined_mask_img


def preprocess_band_files_simple(band_paths):
    band_file_paths = sort_band_paths(band_paths)
    try:
        if not band_file_paths:
            # If no band files are found, return None or any appropriate value
            return None

        band_file_paths = [path for path in band_file_paths if path.endswith(("08.npy", "12.npy", "16.npy"))]
        band_images = [load_band_file(path)[..., 4] for path in band_file_paths]
        band_images = np.stack(band_images, axis=-1)
        band_images = (band_images - np.mean(band_images)) / np.std(band_images)
        return band_images

    except ValueError as e:
        # Handle the ValueError, you can print a message or perform any desired action
        print("Error: No band files found in the subfolder")
        return None


def predict_contrails_avg(model, band_paths):
    band_images = preprocess_band_files_simple(band_paths)
    band_images = np.expand_dims(band_images, axis=0)  # Add a batch dimension

    p0 = model.predict(band_images)[0, ..., 0]  # Original prediction

    # Flip image left-right, make a prediction, then unflip the prediction
    band_images_lr = np.flip(band_images, axis=2)
    p1 = model.predict(band_images_lr)[0, ..., 0]
    p1 = np.flip(p1, axis=1)

    # Flip image up-down, make a prediction, then unflip the prediction
    band_images_ud = np.flip(band_images, axis=1)
    p2 = model.predict(band_images_ud)[0, ..., 0]
    p2 = np.flip(p2, axis=0)

    # Flip image left-right and up-down, make a prediction, then unflip the prediction
    band_images_lr_ud = np.flip(np.flip(band_images, axis=2), axis=1)
    p3 = model.predict(band_images_lr_ud)[0, ..., 0]
    p3 = np.flip(np.flip(p3, axis=1), axis=0)

    # Average the predictions
    prediction_avg = (p0 + p1 + p2 + p3) / 4.0

    return p0, prediction_avg


def visualize_prediction_avg(mask_path, band_paths, model=best_model, figsize=(10, 10), cmap='Spectral', threshold=0.5):
    # Load the mask
    true_mask = np.load(mask_path)[:, :, 0]

    # Get the predicted masks
    p0, predicted_mask_avg = predict_contrails_avg(model, band_paths)

    p0_binary = (p0 > threshold).astype(np.uint8)
    predicted_mask_avg_binary = (predicted_mask_avg > threshold).astype(np.uint8)

    # Visualize band images
    band_vis = visualize_band_images(band_paths)

    # Visualize mask images
    mask_vis = visualize_masks(predicted_mask_avg_binary, true_mask)

    # Print dice coefficient and IoU for original and averaged prediction
    dice_original = dice_coef_pred(true_mask, p0_binary)
    dice_avg = dice_coef_pred(true_mask, predicted_mask_avg_binary)

    jacard_original = jacard_coef_pred(true_mask, p0_binary)
    jacard_avg = jacard_coef_pred(true_mask, predicted_mask_avg_binary)

    print(f"Dice Coefficient - Original: {dice_original}, Averaged: {dice_avg}")
    print(f"IoU - Original: {jacard_original}, Averaged: {jacard_avg}")

    # Prepare the text output
    text_output = f"Dice Coefficient - Averaged: {dice_avg:.3f}\n"
    text_output += f"IoU - Averaged: {jacard_avg:.3f}"

    # Return both outputs
    return band_vis, mask_vis, text_output



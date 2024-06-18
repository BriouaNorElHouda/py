from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image
import rasterio
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.segmentation import slic
from collections import Counter
from matplotlib.patches import Patch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load models
modelRGB = load_model('modelRGB.h5')
modelSegmentation = load_model("model.h5")  # Assuming model.h5 is the segmentation model

# Classes and colors for segmentation
labels = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
class_colors = {
    "AnnualCrop": [118, 205, 38],
    "Forest": [0, 128, 0],
    "HerbaceousVegetation": [255, 255, 0],
    "Highway": [255, 0, 0],
    "Industrial": [128, 0, 128],
    "Pasture": [255, 165, 0],
    "PermanentCrop": [210, 105, 30],
    "Residential": [128, 128, 128],
    "River": [0, 255, 255],
    "SeaLake": [0, 0, 0]
}

# Functions for RGB preprocessing and prediction
def preprocess_image(image_path, target_size=(64, 64)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def model_predict(img_path, modelRGB):
    img = preprocess_image(img_path)
    classes = np.array(['Annual Crop', 'Forest', 'Herbaceous Vegetation', 'Highway', 'Industrial', 'Pasture',
                        'Permanent Crop', 'Residential', 'River', 'Sea Lake'])
    pred = modelRGB.predict(img)
    prediction = np.argmax(pred)
    return classes[prediction]

# Functions for segmentation
def classify_patches(img_array, patch_size=16):
    height, width, _ = img_array.shape
    patch_predictions = np.zeros((height, width), dtype=np.int32)

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = img_array[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue

            patch = image.array_to_img(patch).resize((64, 64))
            patch = image.img_to_array(patch)
            patch = np.expand_dims(patch, axis=0)
            predictions = modelSegmentation.predict(patch)
            predicted_class_index = np.argmax(predictions)
            patch_predictions[i:i+patch_size, j:j+patch_size] = predicted_class_index

    return patch_predictions

def segment_and_merge(image, patch_predictions, original_img_shape):
    resized_patch_predictions = np.zeros(original_img_shape[:2], dtype=np.int32)
    height_ratio = original_img_shape[0] / patch_predictions.shape[0]
    width_ratio = original_img_shape[1] / patch_predictions.shape[1]

    for i in range(original_img_shape[0]):
        for j in range(original_img_shape[1]):
            resized_patch_predictions[i, j] = patch_predictions[int(i / height_ratio), int(j / width_ratio)]

    segments = slic(image, n_segments=100, compactness=10)
    merged_segments = np.zeros_like(segments)

    for seg_val in np.unique(segments):
        mask = (segments == seg_val)
        class_indices, counts = np.unique(resized_patch_predictions[mask], return_counts=True)
        majority_class_index = class_indices[np.argmax(counts)]
        merged_segments[mask] = majority_class_index

    return merged_segments

def plot_results(original_img, merged_segments):
    original_img_array = np.array(original_img)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.axis('off')
    plt.title("Original Image")

    for seg_val in np.unique(merged_segments):
        mask = (merged_segments == seg_val)
        y, x = np.mean(np.argwhere(mask), axis=0)
        class_label = labels[int(seg_val)]
        color = np.array(class_colors[class_label]) / 255

        y_rotated = original_img_array.shape[0] - y
        x_rotated = original_img_array.shape[1] - x

        ha = 'right' if x_rotated > original_img_array.shape[1] / 2 else 'left'
        xytext = (-10, 0) if ha == 'right' else (10, 0)

        plt.annotate(
            '', (x_rotated, y_rotated),
            textcoords="offset points",
            xytext=xytext,
            ha=ha,
            arrowprops=dict(arrowstyle="->", color='white')
        )

        plt.text(
            x_rotated, y_rotated, class_label,
            ha=ha, va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'),
            fontsize=9, color=color
        )

    segmented_image = np.array([class_colors[labels[int(val)]] for val in np.ravel(merged_segments)], dtype=np.uint8)
    segmented_image = segmented_image.reshape((*merged_segments.shape, 3))

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title("Segmented Image")
    plt.axis('off')

    legend_patches = [Patch(color=np.array(color)/255, label=label) for label, color in class_colors.items()]
    plt.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., title="Classes")

    plt.tight_layout()

    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
    plt.savefig(plot_path)
    plt.close()

    counts = Counter(np.ravel(merged_segments))
    for class_index, count in counts.items():
        print(f"{labels[int(class_index)]}: {count} pixels")

    return plot_path

def process_image_rgb(image_path):
    img_array = preprocess_image(image_path)[0]
    patch_predictions = classify_patches(img_array, patch_size=16)
    original_img = image.load_img(image_path)
    original_img_array = image.img_to_array(original_img) / 255.0
    merged_segments = segment_and_merge(original_img_array, patch_predictions, original_img_array.shape)
    plot_path = plot_results(original_img, merged_segments)
    return plot_path

# Function to load and preprocess a single image
def load_and_preprocess_image(img_path):
    with rasterio.open(img_path) as src:
        img_array = src.read()
        img_resized = resize(img_array, (13, 64, 64), mode='reflect', anti_aliasing=True)
        return img_resized

# Function to process the uploaded image and generate result plots
def process_image_multispectral(image_path):
    img_resized = load_and_preprocess_image(image_path)

    # Extract individual bands
    blue_band = img_resized[1, :, :]   # Replace with appropriate band index (0-indexed)
    green_band = img_resized[2, :, :]
    red_band = img_resized[3, :, :]
    nir_band = img_resized[8, :, :]
    swir_band = img_resized[11, :, :]

    # Compute NDVI
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)

    # Compute MNDWI
    mndwi = (green_band - swir_band) / (green_band + swir_band + 1e-8)

    # Compute NDBI
    ndbi = (swir_band - nir_band) / (swir_band + nir_band + 1e-8)

    # Compute BAI
    bai = (1.0 * nir_band - red_band)

    # Compute LSWI
    lswi = (nir_band - swir_band) / (nir_band + swir_band + 1e-8)

    # Compute CI (Chlorophyll Index)
    ci = (nir_band / red_band) - 1

    # Set up the plots in a grid
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Plot NDVI
    im0 = axs[0, 0].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    axs[0, 0].set_title('NDVI (Normalized Difference Vegetation Index)')
    axs[0, 0].axis('off')
    fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    # Plot MNDWI
    im1 = axs[0, 1].imshow(mndwi, cmap='RdYlBu', vmin=-1, vmax=1)
    axs[0, 1].set_title('MNDWI (Modified Normalized Difference Water Index)')
    axs[0, 1].axis('off')
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    # Plot NDBI
    im2 = axs[0, 2].imshow(ndbi, cmap='RdYlBu', vmin=-1, vmax=1)
    axs[0, 2].set_title('NDBI (Normalized Difference Built-up Index)')
    axs[0, 2].axis('off')
    fig.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

    # Plot BAI
    im3 = axs[1, 0].imshow(bai, cmap='RdYlBu', vmin=-1, vmax=1)
    axs[1, 0].set_title('BAI (Burn Area Index) Image')
    axs[1, 0].axis('off')
    fig.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)

    # Plot LSWI
    im4 = axs[1, 1].imshow(lswi, cmap='RdYlBu', vmin=-1, vmax=1)
    axs[1, 1].set_title('LSWI (Land Surface Water Index)')
    axs[1, 1].axis('off')
    fig.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)

    # Plot CI
    im5 = axs[1, 2].imshow(ci, cmap='RdYlBu', vmin=-1, vmax=1)
    axs[1, 2].set_title('CI (Chlorophyll Index)')
    axs[1, 2].axis('off')
    fig.colorbar(im5, ax=axs[1, 2], fraction=0.046, pad=0.04)

    # Adjust layout
    plt.tight_layout()

    # Save the plot to a file
    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
    plt.savefig(plot_path)
    plt.close()

    return plot_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/options')
def options():
    return render_template('selection.html')

@app.route('/predictRGB', methods=['GET', 'POST'])
def uploadRGB():
    if request.method == 'POST':
        if 'imagefileRGB' not in request.files:
            return "No file part"
        imagefileRGB = request.files['imagefileRGB']
        if imagefileRGB.filename == '':
            return "No selected file"
        
        image_path = os.path.join('./images', imagefileRGB.filename)
        imagefileRGB.save(image_path)
        
        prediction_lab = model_predict(image_path, modelRGB)
        
        return render_template('predictRGB.html', predictionRGB=prediction_lab)
    
    return render_template('predictRGB.html')

@app.route('/analyse', methods=['GET', 'POST'])
def analyse():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return redirect(request.url)
        file = request.files['imagefile']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            plot_path = process_image_multispectral(file_path)
            return redirect(url_for('show_result', filename='result.png'))
    return render_template('analyse.html')

@app.route('/segment', methods=['GET', 'POST'])
def segment():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return redirect(request.url)
        file = request.files['imagefile']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            plot_path = process_image_rgb(file_path)
            return redirect(url_for('show_result', filename='result.png'))
    return render_template('segment.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/result')
def show_result():
    filename = request.args.get('filename', 'result.png')
    return render_template('result.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)

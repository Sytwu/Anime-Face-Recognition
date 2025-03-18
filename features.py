import os
import numpy as np
from PIL import Image
import random

# HOG feature extraction (using skimage)
from skimage.feature import hog
from skimage import color

def extract_hog_features(image_path):
    """
    Read an image, convert it to grayscale, and extract HOG features.
    """
    try:
        norm_path = os.path.normpath(image_path)
        image = Image.open(norm_path).convert("RGB")
        image_np = np.array(image)
        gray = color.rgb2gray(image_np)
        features = hog(
            gray,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            feature_vector=True
        )
        return features
    except Exception as e:
        print(f"HOG extract error ({image_path}): {e}")
        return None

# CLIP feature extraction
import torch
import clip

# Load CLIP model and preprocessor (initialized at the first call)
_device = "cuda" if torch.cuda.is_available() else "cpu"
_clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_device)

def extract_clip_features(image_path, model=_clip_model, preprocess=_clip_preprocess):
    """
    Use CLIP's image encoder to convert an image into a feature vector.
    """
    try:
        norm_path = os.path.normpath(image_path)
        image = Image.open(norm_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(_device)
        with torch.no_grad():
            features = model.encode_image(image_input)
        return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"CLIP extract error ({image_path}): {e}")
        return None

# Color-based non-deep learning feature extraction: Color Histogram
def extract_color_histogram(image_path, bins=8):
    """
    Calculate histograms for the RGB channels (each with 8 bins), normalize them,
    and concatenate them into a single feature vector of length 24.
    """
    try:
        norm_path = os.path.normpath(image_path)
        image = Image.open(norm_path).convert("RGB")
        image_np = np.array(image)
        hist_r, _ = np.histogram(image_np[:, :, 0], bins=bins, range=(0, 256))
        hist_g, _ = np.histogram(image_np[:, :, 1], bins=bins, range=(0, 256))
        hist_b, _ = np.histogram(image_np[:, :, 2], bins=bins, range=(0, 256))
        hist = np.concatenate([hist_r, hist_g, hist_b]).astype(float)
        hist /= (hist.sum() + 1e-7)
        return hist
    except Exception as e:
        print(f"Color histogram extract error ({image_path}): {e}")
        return None

# Dictionary mapping feature extraction method names to functions
FEATURE_METHODS = {
    "hog": extract_hog_features,
    "clip": extract_clip_features,
    "color": extract_color_histogram
}

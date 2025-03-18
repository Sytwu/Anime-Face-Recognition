import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# 1. Create InceptionV3 feature extractor by removing the final classification layer
class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        # Set aux_logits to True as required by the pretrained model
        inception = inception_v3(pretrained=True, aux_logits=True)
        # Use selected layers up to the final AdaptiveAvgPool2d layer
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten to [batch, feature_dim]
        return x

# 2. Preprocess image: resize, convert to tensor, and normalize
def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),  # Convert to float tensor and normalize pixel values to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    # Add batch dimension to [1, 3, 299, 299]
    return image_tensor.unsqueeze(0).to(device)

# 3. Extract image feature vectors
def extract_features(model, image_path, device):
    model.eval()
    with torch.no_grad():
        image_tensor = preprocess_image(image_path, device)
        features = model(image_tensor)  # Expected output shape: [1, feature_dim]
    return features.cpu().numpy()  # Keep as 2D array [1, feature_dim]

# 4. Group similar images based on cosine similarity
def group_similar_images(folder, model, device, threshold=0.8):
    # Support webp format as well
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]
    features_dict = {}
    for file in image_files:
        file_path = os.path.join(folder, file)
        try:
            feat = extract_features(model, file_path, device)
            features_dict[file] = feat
        except Exception as e:
            print(f"Error reading {file}: {e}")

    groups = []
    used = set()
    files = list(features_dict.keys())
    for i in range(len(files)):
        if files[i] in used:
            continue
        group = [files[i]]
        used.add(files[i])
        for j in range(i+1, len(files)):
            if files[j] in used:
                continue
            # Use cosine similarity since extract_features returns a 2D array
            sim = cosine_similarity(features_dict[files[i]], features_dict[files[j]])[0][0]
            if sim >= threshold:
                group.append(files[j])
                used.add(files[j])
        groups.append(group)
    return groups

# 5. Display grouped images, aligned horizontally with titles below each image
def display_groups(folder, groups):
    for idx, group in enumerate(groups, start=1):
        if len(group) <= 1:
            continue
        fig, axes = plt.subplots(1, len(group), figsize=(2 * len(group), 2))
        if len(group) == 1:
            axes = [axes]
        fig.suptitle(f"Group {idx}")
        for file in group:
            print(file)
        
        for ax, file in zip(axes, group):
            image_path = os.path.join(folder, file)
            try:
                img = Image.open(image_path).convert('RGB')
                ax.imshow(img)
                ax.set_title(file, fontsize=4, y=-0.12)  # y<0 places title below the image
            except Exception as e:
                print(f"Error reading {file}: {e}")
            ax.axis('off')
        plt.subplots_adjust(wspace=0.15, hspace=0.45)
        plt.show()

if __name__ == '__main__':
    folder_path = 'face_dataset/ame'  # Modify folder path as needed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InceptionV3FeatureExtractor().to(device)
    groups = group_similar_images(folder_path, model, device, threshold=0.99)
    display_groups(folder_path, groups)

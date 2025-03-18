import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

def load_features_labels(csv_file, feature_extraction_func):
    """
    Read a CSV file (which must contain "path" and "label" columns), extract features
    using the provided feature_extraction_func for each row, and return (X, y, paths).
    """
    df = pd.read_csv(csv_file)
    X, y, paths = [], [], []
    for idx, row in df.iterrows():
        feat = feature_extraction_func(row["path"])
        if feat is not None:
            X.append(feat)
            y.append(row["label"])
            paths.append(row["path"])
    return np.array(X), np.array(y), paths

def train_adaboost(X_train, y_train, X_test, y_test):
    """
    Train an AdaBoostClassifier and print the test set accuracy.
    """
    clf = AdaBoostClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("AdaBoost test accuracy:", acc)
    return clf

def train_kmeans(X_train, y_train):
    """
    Train a KMeans clustering model. The number of clusters is set to the number
    of unique labels in the training data.
    """
    num_clusters = len(np.unique(y_train))
    print(f"KMeans number of clusters: {num_clusters}")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X_train)
    print("KMeans clustering completed")
    return kmeans

# -----------------------------
# Deep Learning: ResNet Finetuning (with progress bars, class weighting, and data augmentation options)
# -----------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None, label2idx=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        if label2idx is None:
            labels = self.data["label"].unique()
            self.label2idx = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label2idx = label2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.label2idx[row["label"]]
        return image, label

def train_resnet(train_csv, test_csv, label2idx, num_epochs=5, lr=1e-4, batch_size=32, 
                 num_workers=4, use_class_weight=False, augment=True):
    """
    Finetune a pretrained ResNet18 model and report test set accuracy.
    Displays progress bars for training and evaluation. Adjust settings with
    use_class_weight and augment parameters.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define train transforms based on augmentation option
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageDataset(train_csv, transform=train_transform, label2idx=label2idx)
    test_dataset = ImageDataset(test_csv, transform=test_transform, label2idx=label2idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load a pretrained ResNet18 and replace the final fully connected layer
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(label2idx))
    model = model.to(device)

    # Set up loss function with optional class weighting
    if use_class_weight:
        df_train = pd.read_csv(train_csv)
        classes = np.array(list(label2idx.keys()))
        class_weights = compute_class_weight('balanced', classes=classes, y=df_train['label'])
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        print("Using class weighting:", class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting ResNet finetuning...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for inputs, labels_batch in pbar:
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Evaluation phase
    model.eval()
    correct = 0
    total = 0
    for inputs, labels_batch in tqdm(test_loader, desc="Evaluating", leave=False):
        inputs = inputs.to(device)
        labels_batch = labels_batch.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        total += labels_batch.size(0)
        correct += (preds == labels_batch).sum().item()
    acc = correct / total
    print("ResNet test accuracy:", acc)
    return model

def predict_resnet(model, test_csv, label2idx, batch_size=32, num_workers=4):
    """
    Use the finetuned ResNet model to predict images listed in test_csv.
    Returns a list of (path, pred_label) tuples. The predicted labels are the original class names.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inv_label_map = {v: k for k, v in label2idx.items()}
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_dataset = ImageDataset(test_csv, transform=test_transform, label2idx=label2idx)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    results = []
    model.eval()
    pbar = tqdm(test_loader, desc="Predicting", leave=False)
    all_paths = pd.read_csv(test_csv)["path"].tolist()
    start_idx = 0
    for inputs, _ in pbar:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        batch_size_actual = len(preds)
        batch_paths = all_paths[start_idx:start_idx + batch_size_actual]
        start_idx += batch_size_actual
        for path, pred in zip(batch_paths, preds.cpu().numpy()):
            results.append((path, inv_label_map[pred]))
    return results

def predict_traditional(model, X_test, paths, y_test, method="adaboost"):
    """
    Make predictions with a traditional ML model.
    For KMeans, use a simple mapping based on the assumption that cluster labels roughly correspond to the original labels.
    Returns (preds, accuracy).
    """
    if method == "adaboost":
        preds = model.predict(X_test)
    elif method == "kmeans":
        raw_preds = model.predict(X_test)
        # Build a mapping from clusters to labels. Note: This is a simple example that assumes
        # the cluster label roughly corresponds to the original label. For a robust evaluation,
        # supervised metrics should be used.
        from collections import Counter
        unique_clusters = np.unique(raw_preds)
        mapping = {cluster: str(cluster) for cluster in unique_clusters}
        preds = [mapping[p] for p in raw_preds]
    else:
        preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{method} test accuracy:", acc)
    return preds, acc

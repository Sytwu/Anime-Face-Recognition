import argparse
import pandas as pd
import numpy as np
import os
import random
from features import FEATURE_METHODS
from models import (
    load_features_labels,
    train_adaboost,
    train_kmeans,
    train_resnet,
    predict_resnet,
    predict_traditional,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate models based on the specified feature extraction method and model type"
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="hog",
        choices=["hog", "clip", "resnet", "color", "hog_color"],
        help="Select feature extraction method: hog, clip, resnet, color, hog_color (effective only for traditional ML models)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="adaboost",
        choices=["adaboost", "kmeans", "resnet"],
        help="Select training model: adaboost, kmeans (traditional ML) or resnet (deep learning finetune)",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="train.csv",
        help="Path to training CSV file (used as the full dataset in CV mode)",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="test.csv",
        help="Path to testing CSV file (used in non-CV mode)",
    )
    parser.add_argument(
        "--result_csv",
        type=str,
        default="result.csv",
        help="Path for output prediction results CSV (used in non-CV mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--train_subset_ratio",
        type=float,
        default=1.0,
        help="Training subset ratio (0~1] to experiment with varying amounts of data",
    )
    parser.add_argument(
        "--use_pca",
        type=int,
        default=0,
        help="If greater than 0, apply PCA to reduce traditional model features to the specified number of dimensions",
    )
    parser.add_argument(
        "--use_class_weight",
        action="store_true",
        help="Use class weighting (only applicable for ResNet)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation (only applicable for ResNet; disabled by default)",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=1,
        help="Number of folds for cross-validation (if >1, CV mode is enabled using train_csv as the full dataset)",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_results(results, result_csv):
    """
    Save the results list (each item: (path, pred_label)) to a CSV file.
    """
    df = pd.DataFrame(results, columns=["path", "pred_label"])
    df.to_csv(result_csv, index=False)
    print(f"Prediction results have been saved to {result_csv}")


def run_traditional_model_cv(feature_func, model_name, df_all, args):
    """
    Traditional ML models use StratifiedKFold CV.
    For each fold, the output filename is determined based on the result_csv parameter.
    For example, if result_csv = XXXX.csv, then the output for fold i will be XXXX_fold{i}.csv.
    """
    X_all, y_all, paths_all = load_features_labels(args.train_csv, feature_func)

    if args.train_subset_ratio < 1.0:
        total = X_all.shape[0]
        idx = np.random.choice(total, int(total * args.train_subset_ratio), replace=False)
        X_all = X_all[idx]
        y_all = np.array(y_all)[idx]
        paths_all = [paths_all[i] for i in idx]
        print(f"Using {len(y_all)} samples (originally {total} samples)")

    if args.use_pca > 0:
        print(f"Using PCA to reduce dimensions to {args.use_pca}")
        pca = PCA(n_components=args.use_pca)
        X_all = pca.fit_transform(X_all)
        print("PCA explained variance ratio:", pca.explained_variance_ratio_)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold = 1

    for train_idx, val_idx in skf.split(X_all, y_all):
        print(f"=== Fold {fold} ===")
        X_train, y_train = X_all[train_idx], np.array(y_all)[train_idx]
        X_val, y_val = X_all[val_idx], np.array(y_all)[val_idx]
        paths_val = [paths_all[i] for i in val_idx]

        if model_name == "adaboost":
            model = train_adaboost(X_train, y_train, X_val, y_val)
            preds = model.predict(X_val)
        elif model_name == "kmeans":
            model = train_kmeans(X_train, y_train)
            raw_preds = model.predict(X_val)
            # Map clusters to labels based on majority voting
            from collections import Counter

            unique_clusters = np.unique(raw_preds)
            mapping = {}
            for cluster in unique_clusters:
                indices = np.where(raw_preds == cluster)[0]
                labels_in_cluster = y_val[indices]
                mapping[cluster] = Counter(labels_in_cluster).most_common(1)[0][0]
            preds = [mapping[p] for p in raw_preds]
        else:
            raise ValueError(f"Unknown model: {model_name}")

        acc = accuracy_score(y_val, preds)
        print(f"Fold {fold} Accuracy: {acc}")

        # Determine result filename based on result_csv parameter
        if args.result_csv:
            base_name = os.path.splitext(os.path.basename(args.result_csv))[0]
            result_csv = f"{base_name}_fold{fold}.csv"
        else:
            result_csv = f"result_fold{fold}.csv"

        results = list(zip(paths_val, preds))
        save_results(results, result_csv)
        fold += 1


def run_resnet_cv(label2idx, args):
    """
    ResNet CV mode: use StratifiedKFold to split train_csv,
    apply train_subset_ratio to each training fold, and save each fold's results.
    For example, if result_csv = XXXX.csv, then each fold is saved as XXXX_fold{i}.csv.
    """
    df_all = pd.read_csv(args.train_csv)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold = 1

    for train_idx, val_idx in skf.split(df_all, df_all["label"]):
        print(f"=== Fold {fold} ===")
        train_fold = df_all.iloc[train_idx]
        val_fold = df_all.iloc[val_idx]

        # Subsample training fold if needed
        if args.train_subset_ratio < 1.0:
            total = len(train_fold)
            train_fold = train_fold.sample(frac=args.train_subset_ratio, random_state=args.seed)
            print(f"Fold {fold}: Using {len(train_fold)} training samples (originally {total} samples)")

        temp_train_csv = f"temp_train_fold{fold}.csv"
        temp_val_csv = f"temp_val_fold{fold}.csv"
        train_fold.to_csv(temp_train_csv, index=False)
        val_fold.to_csv(temp_val_csv, index=False)

        labels = train_fold["label"].unique()
        local_label2idx = {label: idx for idx, label in enumerate(labels)}
        model = train_resnet(temp_train_csv, temp_val_csv, local_label2idx,
                             use_class_weight=args.use_class_weight, augment=args.augment)
        results = predict_resnet(model, temp_val_csv, local_label2idx)

        df_val = pd.read_csv(temp_val_csv)
        df_pred = pd.DataFrame(results, columns=["path", "pred_label"])
        df_merged = pd.merge(df_val, df_pred, on="path", how="inner")
        acc = accuracy_score(df_merged["label"], df_merged["pred_label"])
        print(f"Fold {fold} ResNet test accuracy: {acc}")

        if args.result_csv:
            base_name = os.path.splitext(os.path.basename(args.result_csv))[0]
            result_csv = f"{base_name}_fold{fold}.csv"
        else:
            result_csv = f"result_fold{fold}.csv"

        save_results(results, result_csv)
        os.remove(temp_train_csv)
        os.remove(temp_val_csv)
        fold += 1


def run_single_mode(feature_func, label2idx, args):
    """
    Non-CV mode: run traditional ML or ResNet training based on the model parameter.
    """
    if args.model in ["adaboost", "kmeans"]:
        print(f"Using {args.feature} for feature extraction and training {args.model}...")
        X_train, y_train, _ = load_features_labels(args.train_csv, feature_func)
        X_test, y_test, paths = load_features_labels(args.test_csv, feature_func)

        if args.train_subset_ratio < 1.0:
            total = X_train.shape[0]
            idx = np.random.choice(total, int(total * args.train_subset_ratio), replace=False)
            X_train = X_train[idx]
            y_train = np.array(y_train)[idx]
            print(f"Using {len(y_train)} training samples (originally {total} samples)")

        if args.use_pca > 0:
            print(f"Using PCA to reduce dimensions to {args.use_pca}")
            pca = PCA(n_components=args.use_pca)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            print("PCA explained variance ratio:", pca.explained_variance_ratio_)

        if args.model == "adaboost":
            model = train_adaboost(X_train, y_train, X_test, y_test)
            preds = model.predict(X_test)
        elif args.model == "kmeans":
            model = train_kmeans(X_train, y_train)
            raw_preds = model.predict(X_test)
            from collections import Counter

            unique_clusters = np.unique(raw_preds)
            mapping = {}
            for cluster in unique_clusters:
                indices = np.where(raw_preds == cluster)[0]
                labels_in_cluster = np.array(y_test)[indices]
                mapping[cluster] = Counter(labels_in_cluster).most_common(1)[0][0]
            preds = [mapping[p] for p in raw_preds]

        acc = accuracy_score(y_test, preds)
        print("Traditional ML model test accuracy:", acc)
        results = list(zip(paths, preds))
        save_results(results, args.result_csv)

    elif args.model == "resnet":
        print("Using deep learning ResNet finetune for training and prediction...")
        train_df = pd.read_csv(args.train_csv)

        # Subsample training data if train_subset_ratio < 1.0
        if args.train_subset_ratio < 1.0:
            total = len(train_df)
            train_df = train_df.sample(frac=args.train_subset_ratio, random_state=args.seed)
            print(f"Using {len(train_df)} training samples (originally {total} samples)")

        labels = train_df["label"].unique()
        local_label2idx = {label: idx for idx, label in enumerate(labels)}

        # Save the subsampled training data temporarily
        temp_train_csv = "temp_train_resnet.csv"
        train_df.to_csv(temp_train_csv, index=False)
        model = train_resnet(temp_train_csv, args.test_csv, local_label2idx,
                             use_class_weight=args.use_class_weight, augment=args.augment)
        results = predict_resnet(model, args.test_csv, local_label2idx)

        # Calculate accuracy using test CSV
        test_df = pd.read_csv(args.test_csv)
        df_pred = pd.DataFrame(results, columns=["path", "pred_label"])
        df_merged = pd.merge(test_df, df_pred, on="path", how="inner")
        acc = accuracy_score(df_merged["label"], df_merged["pred_label"])
        print("ResNet test accuracy:", acc)
        os.remove(temp_train_csv)
        save_results(results, args.result_csv)

    else:
        raise ValueError(f"Unknown model selection: {args.model}")


def main():
    args = parse_args()
    set_seed(args.seed)

    # For non-CV mode, set the feature extraction method (only applicable to traditional ML models)
    feature_func = FEATURE_METHODS[args.feature]

    if args.n_folds > 1:
        print(f"Enabling {args.n_folds}-fold cross validation using {args.train_csv} as the full dataset.")
        df_all = pd.read_csv(args.train_csv)
        if args.model in ["adaboost", "kmeans"]:
            run_traditional_model_cv(feature_func, args.model, df_all, args)
        elif args.model == "resnet":
            run_resnet_cv({}, args)  # Label mapping will be created within each fold
        else:
            raise ValueError(f"Unknown model selection: {args.model}")
    else:
        # Non-CV mode
        run_single_mode(feature_func, {}, args)


if __name__ == "__main__":
    main()

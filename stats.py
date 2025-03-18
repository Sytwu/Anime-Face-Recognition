import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare and visualize test.csv with the prediction result CSV."
    )
    parser.add_argument(
        '--test_csv',
        type=str,
        default='test.csv',
        help="Path to the original test CSV (must contain 'path' and 'label' columns)."
    )
    parser.add_argument(
        '--result_csv',
        type=str,
        default='result.csv',
        help="Path to the prediction results CSV (must contain 'path' and 'pred_label' columns)."
    )
    parser.add_argument(
        '--out_fig',
        type=str,
        default='confusion_matrix.png',
        help="Output path for the confusion matrix image."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Read the test data and prediction results
    df_test = pd.read_csv(args.test_csv)
    df_result = pd.read_csv(args.result_csv)
    
    # Merge data based on the 'path' column
    df = pd.merge(df_test, df_result, on="path", how="inner")
    
    # Check if the merged DataFrame contains the required 'label' and 'pred_label' columns
    if "label" not in df.columns or "pred_label" not in df.columns:
        print("CSV column error. Please ensure test.csv contains 'label' and result.csv contains 'pred_label'.")
        return

    y_true = df["label"]
    y_pred = df["pred_label"]
    
    # Compute the confusion matrix
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(args.out_fig)
    plt.show()
    
    # Print the detailed classification report
    report = classification_report(y_true, y_pred, labels=labels)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    main()

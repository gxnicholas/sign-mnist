import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def main():
    """Main driver code"""
    train_df = pd.read_csv("../data/sign_mnist_train.csv")
    test_df = pd.read_csv("../data/sign_mnist_test.csv")

    # Remove labels
    train_features, test_features = train_df.iloc[:, 1:], test_df.iloc[:, 1:]
    train_labels, test_labels = train_df.iloc[:, 0], test_df.iloc[:, 0]

    # Training, they make it so easy in 2021
    print("Training with the 2 line wonder")
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train_features, train_labels)

    # Prediction and results
    pred = rf.predict(test_features)
    print("Classification Report")
    print(classification_report(test_labels, pred))
    cm = confusion_matrix(test_labels, pred)
    cm_df = pd.DataFrame(
        cm,
        index=[i for i in "ABCDEFGHIKLMNOPQRSTUVWXY"],
        columns=[i for i in "ABCDEFGHIKLMNOPQRSTUVWXY"],
    )
    plt.figure(figsize=(10, 7))
    sn.heatmap(cm_df, annot=True)
    plt.show()


if __name__ == "__main__":
    main()

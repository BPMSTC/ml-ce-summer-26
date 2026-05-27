from pathlib import Path
import os
import warnings

# This tiny demo does not need parallel workers. Keeping joblib to one worker
# also avoids a noisy Windows physical-core detection warning on some systems.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores*",
    category=UserWarning,
    module="joblib.externals.loky.backend.context",
)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    adjusted_rand_score,
    classification_report,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42


def load_iris_table() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the small built-in Iris dataset as a feature table plus labels."""
    iris = load_iris(as_frame=True)
    features = iris.data
    labels = iris.target
    target_names = [str(name) for name in iris.target_names]

    print("Loaded the Iris dataset.")
    print(f"Rows: {features.shape[0]}")
    print(f"Input columns: {list(features.columns)}")
    print(f"Known species labels: {target_names}")

    return features, labels, target_names


def run_supervised_demo(
    features: pd.DataFrame, labels: pd.Series, target_names: list[str]
) -> None:
    """Train a classifier that learns from examples where the answer is known."""
    print("\n" + "=" * 80)
    print("SUPERVISED LEARNING DEMO")
    print("=" * 80)
    print(
        "Goal: learn a mapping from flower measurements to known species labels.\n"
        "The model is allowed to see the correct answer during training."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy on held-out test flowers: {accuracy:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=target_names))

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=target_names,
        cmap="Blues",
        xticks_rotation=35,
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Supervised Demo: Predicted Species")
    fig.tight_layout()
    output_path = OUTPUT_DIR / "iris_supervised_confusion_matrix.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved confusion matrix: {output_path}")


def run_unsupervised_demo(
    features: pd.DataFrame, labels: pd.Series, target_names: list[str]
) -> None:
    """Cluster flowers without giving the model the species labels."""
    print("\n" + "=" * 80)
    print("UNSUPERVISED LEARNING DEMO")
    print("=" * 80)
    print(
        "Goal: find natural groups using only the measurements.\n"
        "The model is not given the species labels while it learns."
    )

    clustering_model = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    clusters = clustering_model.fit_predict(features)

    comparison = pd.DataFrame(
        {
            "species": labels.map(lambda index: target_names[index]),
            "cluster": clusters,
        }
    )

    print("\nCluster counts:")
    print(comparison["cluster"].value_counts().sort_index())

    print(
        "\nAfter clustering, we can compare clusters to the real species labels.\n"
        "This comparison is for learning and evaluation only. The labels were not used by KMeans."
    )
    print(pd.crosstab(comparison["cluster"], comparison["species"]))

    agreement = adjusted_rand_score(labels, clusters)
    print(f"\nCluster/species agreement score: {agreement:.3f}")
    print(
        "A score near 1.0 means the discovered groups line up well with the known labels.\n"
        "A score near 0.0 means the grouping is no better than random."
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        features["petal length (cm)"],
        features["petal width (cm)"],
        c=clusters,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_title("Unsupervised Demo: KMeans Clusters")
    ax.set_xlabel("Petal length (cm)")
    ax.set_ylabel("Petal width (cm)")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "iris_unsupervised_clusters.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved cluster plot: {output_path}")


def main() -> None:
    print("Supervised vs. unsupervised learning with the Iris dataset")
    print(f"RANDOM_STATE={RANDOM_STATE} makes the demo repeatable.")

    features, labels, target_names = load_iris_table()

    print("\nFirst five rows of input measurements:")
    print(features.head())

    run_supervised_demo(features, labels, target_names)
    run_unsupervised_demo(features, labels, target_names)

    print("\nKey difference:")
    print(
        "Supervised learning trains with answers attached to examples. "
        "Unsupervised learning trains without answers and looks for structure."
    )


if __name__ == "__main__":
    main()

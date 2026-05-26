from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "support_tickets.csv"
RANDOM_STATE = 42


def load_english_tickets(data_path: Path) -> pd.DataFrame:
    """Load support tickets and keep the English-language rows used in the workshop."""
    tickets = pd.read_csv(data_path)

    english_tickets = tickets[tickets["language"].str.lower() == "en"].copy()
    english_tickets["subject"] = english_tickets["subject"].fillna("")
    english_tickets["body"] = english_tickets["body"].fillna("")
    english_tickets["ticket_text"] = (
        english_tickets["subject"] + "\n\n" + english_tickets["body"]
    ).str.strip()

    usable_tickets = english_tickets[
        (english_tickets["ticket_text"] != "") & english_tickets["queue"].notna()
    ].copy()

    return usable_tickets


def build_model() -> Pipeline:
    """Create a beginner-friendly text classification pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    tickets = load_english_tickets(DATA_PATH)

    print(f"Loaded {len(tickets):,} English-language tickets.")
    print("\nSupport queue counts:")
    print(tickets["queue"].value_counts())

    X = tickets["ticket_text"]
    y = tickets["queue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    labels = list(model.classes_)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    confusion = pd.DataFrame(matrix, index=labels, columns=labels)

    print("\nConfusion matrix:")
    print(confusion)

    results = pd.DataFrame(
        {
            "text": X_test,
            "actual_queue": y_test,
            "predicted_queue": predictions,
        }
    )
    mistakes = results[results["actual_queue"] != results["predicted_queue"]]

    print("\nA few misclassified examples:")
    for _, row in mistakes.head(5).iterrows():
        preview = row["text"].replace("\n", " ")[:240]
        print("-" * 80)
        print(f"Actual:    {row['actual_queue']}")
        print(f"Predicted: {row['predicted_queue']}")
        print(f"Text:      {preview}...")

    custom_examples = [
        "I was charged twice for my monthly subscription and need help with my invoice.",
        "The application is unavailable for all users and our team cannot access the portal.",
        "Can you explain whether this product integrates with our CRM and analytics tools?",
    ]

    print("\nCustom example predictions:")
    for example, prediction in zip(custom_examples, model.predict(custom_examples)):
        print("-" * 80)
        print(f"Text:      {example}")
        print(f"Predicted: {prediction}")


if __name__ == "__main__":
    main()

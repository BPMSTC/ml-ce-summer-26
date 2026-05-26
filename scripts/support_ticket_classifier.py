from pathlib import Path
from time import perf_counter

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
    print("Step 2 started: loading the support ticket CSV...", flush=True)

    start = perf_counter()
    tickets = pd.read_csv(data_path)
    elapsed = perf_counter() - start

    print(
        f"Step 2 complete: loaded {tickets.shape[0]:,} rows and "
        f"{tickets.shape[1]:,} columns in {elapsed:.2f} seconds.",
        flush=True,
    )

    print("Step 4 started: filtering to English-language tickets...", flush=True)

    english_tickets = tickets[tickets["language"].str.lower() == "en"].copy()

    print("Step 5 started: preparing text features and queue labels...", flush=True)

    # Text models cannot use missing text directly, so blank subject/body values
    # become empty strings. We keep the row usable if one of the fields exists.
    english_tickets["subject"] = english_tickets["subject"].fillna("")
    english_tickets["body"] = english_tickets["body"].fillna("")

    # The subject and body are available when a new ticket arrives. The answer
    # column is intentionally excluded because it would leak future information.
    english_tickets["ticket_text"] = (
        english_tickets["subject"] + "\n\n" + english_tickets["body"]
    ).str.strip()

    usable_tickets = english_tickets[
        (english_tickets["ticket_text"] != "") & english_tickets["queue"].notna()
    ].copy()

    print(
        f"Step 5 complete: prepared {len(usable_tickets):,} usable English tickets.",
        flush=True,
    )

    return usable_tickets


def build_model() -> Pipeline:
    """Create a beginner-friendly text classification pipeline."""
    print("Step 7 started: building the TF-IDF plus Logistic Regression pipeline...", flush=True)

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    # Removes common English words such as "the" and "and".
                    stop_words="english",
                    # Keeps the vocabulary bounded so training remains fast.
                    max_features=10000,
                    # Uses individual words and two-word phrases.
                    ngram_range=(1, 2),
                    # Ignores terms that appear only once because they rarely generalize.
                    min_df=2,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    # Gives the optimizer enough iterations to converge.
                    max_iter=2000,
                    # Helps smaller queue classes influence training.
                    class_weight="balanced",
                    # Makes any internal randomized behavior repeatable.
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    print("Step 1 complete: imports loaded and constants configured.", flush=True)
    print(
        "Runtime note: this script uses CPU-based scikit-learn, not the GPU.",
        flush=True,
    )
    print(
        "RANDOM_STATE=42 is a repeatability seed, not a mathematically special value.",
        flush=True,
    )

    tickets = load_english_tickets(DATA_PATH)

    print(f"Loaded {len(tickets):,} English-language tickets.")
    print("\nSupport queue counts:")
    print(tickets["queue"].value_counts())

    X = tickets["ticket_text"]
    y = tickets["queue"]

    print("Step 6 started: splitting rows into training and test sets...", flush=True)

    # We do not train on all rows because that would hide whether the model can
    # generalize. The test set estimates behavior on unseen tickets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"Step 6 complete: {len(X_train):,} training rows and {len(X_test):,} test rows.",
        flush=True,
    )

    model = build_model()
    print("Step 7 complete: pipeline created.", flush=True)

    print("Step 8 started: training the model. This may take a little while...", flush=True)
    start = perf_counter()
    model.fit(X_train, y_train)
    elapsed = perf_counter() - start
    print(f"Step 8 complete: model training finished in {elapsed:.2f} seconds.", flush=True)

    print("Step 9 started: evaluating the model on held-out test rows...", flush=True)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.3f}")
    print("\nClassification report:")
    print("precision: predicted as this queue and actually correct")
    print("recall: actual tickets in this queue that the model found")
    print("f1-score: balance between precision and recall")
    print("support: number of test examples for that queue")
    print(classification_report(y_test, predictions))

    labels = list(model.classes_)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    confusion = pd.DataFrame(matrix, index=labels, columns=labels)

    print("\nConfusion matrix:")
    print(confusion)
    print("Step 9 complete: evaluation metrics generated.", flush=True)

    print("Step 10 started: finding examples where the model was wrong...", flush=True)
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

    print("Step 10 complete: misclassified examples printed.", flush=True)

    print("Step 11 started: scoring custom examples...", flush=True)
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

    print("Step 11 complete: custom examples scored.", flush=True)


if __name__ == "__main__":
    main()

# Applied Machine Learning Workshop

This project contains the hands-on materials for a 3-hour continuing education workshop on applied machine learning with Python.

The workshop uses a customer support ticket dataset to build a simple text classifier that predicts the support queue for an incoming ticket. The goal is not to build a production system during the session. The goal is to understand the applied machine learning workflow: inspect data, prepare features and labels, train a baseline model, evaluate it, and discuss responsible use.

## Project Files

- `data/support_tickets.csv` - local copy of the support ticket dataset.
- `notebooks/ml_workshop_support_ticket_classification.ipynb` - fully worked Jupyter notebook for the live workshop.
- `notebooks/supervised_unsupervised_iris_demo.ipynb` - short teaching demo that contrasts supervised classification with unsupervised clustering.
- `scripts/support_ticket_classifier.py` - plain Python version of the notebook workflow.
- `scripts/supervised_unsupervised_iris_demo.py` - plain Python version of the supervised vs. unsupervised Iris demo.
- `requirements.txt` - Python dependencies.
- `setup_env.ps1` - PowerShell setup script that creates the virtual environment and registers the Jupyter kernel.

## Dataset Source

The support ticket dataset used in this workshop is a local copy of this Hugging Face dataset file:

[Tobi-Bueck/customer-support-tickets: aa_dataset-tickets-multi-lang-5-2-50-version.csv](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/blob/main/aa_dataset-tickets-multi-lang-5-2-50-version.csv)

The local project copy is stored at:

```text
data/support_tickets.csv
```

The notebook filters this dataset to English-language rows and uses `queue` as the prediction target.

## Setup: PowerShell

From the project root, run:

```powershell
.\setup_env.ps1
```

If PowerShell blocks the script because of execution policy, run this command for the current terminal session and then try again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\setup_env.ps1
```

The setup script will:

1. Create a virtual environment in `.venv`.
2. Install the packages in `requirements.txt`.
3. Register a Jupyter kernel named `ml-workshop` with the display name `Python (ML Workshop)`.

## Start Jupyter

After setup completes, start Jupyter with:

```powershell
.\.venv\Scripts\jupyter notebook
```

Open:

```text
notebooks/ml_workshop_support_ticket_classification.ipynb
```

For the short supervised vs. unsupervised learning demo, open:

```text
notebooks/supervised_unsupervised_iris_demo.ipynb
```

Select this kernel if it is not already selected:

```text
Python (ML Workshop)
```

## Optional Conda Setup

The recommended setup is the built-in Python `venv` flow above. If you prefer Conda, this project can also run in a Conda environment:

```powershell
conda create -n ml-workshop python=3.11
conda activate ml-workshop
pip install -r requirements.txt
python -m ipykernel install --user --name ml-workshop --display-name "Python (ML Workshop)"
jupyter notebook
```

For this workshop, `venv` is simpler because the dependencies are standard Python data science packages and do not require GPU tooling or complex system libraries.

## Workshop Flow

The main support-ticket notebook walks through:

1. Loading the dataset.
2. Filtering to English-language tickets.
3. Inspecting columns, missing values, and class balance.
4. Creating text features from the ticket subject and body.
5. Choosing `queue` as the prediction target.
6. Splitting into training and test sets.
7. Training a TF-IDF plus Logistic Regression baseline model.
8. Evaluating accuracy, precision, recall, F1 score, and a confusion matrix.
9. Inspecting model mistakes.
10. Trying custom ticket examples.
11. Discussing responsible use and production considerations.

## Short Demo: Supervised vs. Unsupervised Learning

Before the main support-ticket classifier, use this shorter notebook:

```text
notebooks/supervised_unsupervised_iris_demo.ipynb
```

It uses the built-in Iris flower dataset from scikit-learn to compare two learning styles:

- **Supervised learning:** a Logistic Regression classifier learns from flower measurements with known species labels.
- **Unsupervised learning:** a KMeans clustering model uses the same measurements but does not receive the species labels during training.

The demo is intentionally small and heavily explained. It walks through:

1. The difference between input features `X` and answer labels `y`.
2. Why supervised learning trains with an answer key.
3. How a train/test split helps evaluate predictions on unseen examples.
4. How to read accuracy, precision, recall, F1 score, and a confusion matrix.
5. Why unsupervised learning trains without labels.
6. How KMeans clustering discovers groups from measurements.
7. Why cluster numbers are not the same thing as class labels.
8. How cluster results can be compared to known labels after the fact for teaching purposes.

The purpose is to make the conceptual difference clear before learners move into the more realistic support-ticket classification workflow.

## Run the Script Version

The notebooks are the main workshop artifacts. Script versions are included as compact references.

Run the support-ticket classifier script:

```powershell
.\.venv\Scripts\python scripts\support_ticket_classifier.py
```

Run the supervised vs. unsupervised Iris demo script:

```powershell
.\.venv\Scripts\python scripts\supervised_unsupervised_iris_demo.py
```

The Iris script saves two generated figures under `outputs/`, which is ignored by Git.

## Notes

This workshop uses English-language rows only. The original dataset also includes German-language tickets, which are useful for a later discussion about multilingual data and production complexity.

The image/logo normalization topic is intentionally left as a future extension. It involves different data preparation, modeling, and evaluation workflows than this text classification exercise.

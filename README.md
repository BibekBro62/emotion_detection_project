# Emotion Detector

A Streamlit app that predicts an emotion from text using the included TF-IDF vectorizer and logistic-regression model.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Running Streamlit as a Python module avoids PowerShell's `streamlit` command-not-found error when the Python Scripts folder is not on `PATH`.

The app expects these files in `model/`:

- `tfidf_vectorizer.pkl`
- `logistic_regression_model.pkl`

## Class labels

The model artifact contains numeric classes but does not include the mapping from class IDs to emotion names. The app defaults to `0=Sadness`, `1=Anger`, `2=Love`, `3=Surprise`, `4=Fear`, and `5=Joy`, and provides an editable mapping in the sidebar. Confirm this matches the label-to-ID mapping used when training the model.

Only load pickle files you trust. Pickle can execute code while loading. The model artifacts were serialized with scikit-learn 1.6.1, so `requirements.txt` pins that version to avoid cross-version pickle warnings and compatibility risks.

## Deploy to Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and both files in `model/` to a GitHub repository.
2. Create a new app in Streamlit Community Cloud and select that repository, branch, and `app.py` as the main file.
3. Deploy. Streamlit Cloud installs the packages from `requirements.txt` before starting the app.
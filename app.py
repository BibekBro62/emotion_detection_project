from pathlib import Path
import pickle

import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
EMOTIONS = ("Sadness", "Joy", "Love", "Anger", "Fear", "Surprise")
DEFAULT_LABELS = {
    0: "Sadness",
    1: "Anger",
    2: "Love",
    3: "Surprise",
    4: "Fear",
    5: "Joy",
}
EXAMPLES = {
    "Sadness": "I didnt feel humiliated",
    "Anger": "I feel so angry I could scream",
    "Fear": "I am afraid to watch the horror movie",
    "Joy": "I feel happy and excited about my future",
    "Love": "i was ready to meet mom in the airport and feel her ever supportive arms around me",
    "Surprise": "I am surprised by the news",
}


st.set_page_config(
    page_title="Emotion analysis",
    page_icon="💬",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    :root {
        color-scheme: light;
        --ink: #1d302c;
        --muted: #687873;
        --paper: #f6f8f6;
        --accent: #187a66;
        --line: #dce5df;
        --font-ui: 'DM Sans', 'Segoe UI', sans-serif;
    }
    html, body, [class*="css"], button, input, textarea {
        font-family: var(--font-ui);
    }
    [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container {
        max-width: 1120px;
        padding: 2.5rem clamp(1rem, 4vw, 3.25rem) 4rem;
    }
    h1, h2, h3, p, label { color: var(--ink); letter-spacing: 0; }
    h1 { font-size: 2.25rem; font-weight: 700; }
    h2, h3 { font-weight: 600; }
    .eyebrow {
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .lede { color: var(--muted); font-size: 1rem; margin-top: -0.5rem; }
    [data-testid="stSidebar"] {
        background: #eef3ef;
        border-right: 1px solid var(--line);
    }
    [data-testid="stTextArea"] textarea { min-height: 150px; }
    [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stTextArea"] textarea { border-radius: 8px; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 8px;
        background: #fff;
    }
    div.stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primaryFormSubmit"] {
        background: var(--accent);
        border-color: var(--accent);
        color: white;
        font-weight: 600;
    }
    div[data-testid="stProgressBarTrack"] > div { background: var(--accent); }
    @media (max-width: 640px) {
        .block-container { padding: 1.5rem 1rem 2.5rem; }
        h1 { font-size: 1.8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    with (MODEL_DIR / "tfidf_vectorizer.pkl").open("rb") as file:
        vectorizer = pickle.load(file)
    with (MODEL_DIR / "logistic_regression_model.pkl").open("rb") as file:
        model = pickle.load(file)
    return vectorizer, model


st.markdown('<div class="eyebrow">Text classification</div>', unsafe_allow_html=True)
st.title("Emotion analysis")
st.markdown(
    '<p class="lede">Classify a sentence with your TF-IDF and logistic regression model.</p>',
    unsafe_allow_html=True,
)

try:
    vectorizer, model = load_artifacts()
except FileNotFoundError as error:
    st.error(f"Model file not found: {error.filename}. Keep both pickle files in the model folder.")
    st.stop()
except Exception as error:
    st.error(f"Could not load the model artifacts: {error}")
    st.stop()

class_ids = [int(class_id) for class_id in model.classes_]
label_options = ("Unmapped", *EMOTIONS)

with st.sidebar:
    st.subheader("Class labels")
    st.caption("The model stores numeric classes. Confirm this mapping matches your training labels.")
    class_labels = {}
    for class_id in class_ids:
        default_label = DEFAULT_LABELS.get(class_id, "Unmapped")
        default_index = label_options.index(default_label)
        label_key = f"class_label_{class_id}"
        if label_key in st.session_state:
            if st.session_state[label_key] == "Unmapped":
                st.session_state[label_key] = default_label
            class_labels[class_id] = st.selectbox(
                f"Class {class_id}", label_options, key=label_key
            )
        else:
            class_labels[class_id] = st.selectbox(
                f"Class {class_id}",
                label_options,
                index=default_index,
                key=label_key,
            )

with st.container(border=True):
    st.subheader("Analyze text")
    sample_column, button_column = st.columns([3, 1], vertical_alignment="bottom")
    with sample_column:
        sample_choice = st.selectbox("Example sentence", ("Choose an example", *EXAMPLES.keys()))
    with button_column:
        if st.button("Use example", disabled=sample_choice == "Choose an example", use_container_width=True):
            st.session_state["text_input"] = EXAMPLES[sample_choice]

    with st.form("emotion_form"):
        text = st.text_area(
            "Your text",
            placeholder="Write or paste a sentence to analyze...",
            height=150,
            key="text_input",
        )
        submitted = st.form_submit_button("Predict emotion", type="primary", use_container_width=True)

if submitted:
    if not text.strip():
        st.warning("Enter some text before running a prediction.")
    else:
        try:
            features = vectorizer.transform([text.strip()])
            probabilities = model.predict_proba(features)[0]
            predicted_class = int(model.classes_[probabilities.argmax()])
            st.session_state["prediction"] = {
                "text": text.strip(),
                "class_id": predicted_class,
                "probabilities": probabilities.tolist(),
            }
        except Exception as error:
            st.error(f"Prediction failed: {error}")

prediction = st.session_state.get("prediction")
if prediction:
    predicted_label = class_labels.get(prediction["class_id"], "Unmapped")
    if predicted_label == "Unmapped":
        predicted_label = f"Class {prediction['class_id']}"
    confidence = max(prediction["probabilities"])
    with st.container(border=True):
        result_column, confidence_column = st.columns([3, 1])
        with result_column:
            st.markdown('<div class="eyebrow">Prediction</div>', unsafe_allow_html=True)
            st.subheader(predicted_label)
        with confidence_column:
            st.metric("Confidence", f"{confidence:.1%}")

        st.markdown("**Class probabilities**")
        probability_columns = st.columns(2)
        for index, (class_id, probability) in enumerate(zip(class_ids, prediction["probabilities"])):
            label = class_labels.get(class_id, "Unmapped")
            display_label = label if label != "Unmapped" else f"Class {class_id}"
            with probability_columns[index % 2]:
                st.write(f"{display_label} · {probability:.1%}")
                st.progress(float(probability))
        st.caption("Probabilities are model scores, not a guarantee that an emotion is present.")
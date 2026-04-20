import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from groq import Groq
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="AI-NIDS Student Project", layout="wide")

st.title("AI-Based Network Intrusion Detection System")
st.markdown("""
*Student Project: This system uses **Random Forest* to detect Network attacks and *Groq AI* to explain the packets.
""")

# --- SIDEBAR: SETTINGS ---
st.sidebar.header("0. Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help=r"C:\Users\ravid\Downloads\archive (5)\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

st.sidebar.header("1. Settings")
groq_api_key = st.sidebar.text_input("Groq API Key (starts with gsk_)", type="password")
st.sidebar.caption("[Get a free key here](https://console.groq.com/keys)")

st.sidebar.header("2. Model Training")

# --- DATA LOADING ---
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file, nrows=15000)
        df.columns = df.columns.str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# --- MODEL TRAINING ---
def train_model(df):
    features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Fwd Packet Length Max',
        'Flow IAT Mean', 'Flow IAT Std', 'Flow Packets/s'
    ]
    target = 'Label'

    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in CSV: {missing_cols}")
        return None, 0, [], None, None

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    score = accuracy_score(y_test, clf.predict(X_test))

    # Save model
    joblib.dump(clf, 'model.pkl')

    return clf, score, features, X_test, y_test

# --- LOAD SAVED MODEL IF EXISTS ---
if 'model' not in st.session_state and os.path.exists('model.pkl'):
    st.session_state['model'] = joblib.load('model.pkl')
    st.sidebar.info("Saved model loaded automatically.")

# --- APP LOGIC ---
if uploaded_file is not None:
    df = load_data(uploaded_file)

    if df is not None:
        st.sidebar.success(f"Dataset Loaded: {len(df)} rows")

        if st.sidebar.button("Train Model Now"):
            with st.spinner("Training model... please wait"):
                clf, accuracy, feature_names, X_test, y_test = train_model(df)
                if clf:
                    st.session_state['model'] = clf
                    st.session_state['features'] = feature_names
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    st.session_state['accuracy'] = accuracy
                    st.sidebar.success(f"Training Complete! Accuracy: {accuracy:.2%}")
else:
    st.warning("Please upload your CSV file in the sidebar to begin.")
    st.stop()

# --- TABS ---
tab1, tab2, tab3 = st.tabs([
    "Threat Analysis",
    "Model Evaluation",
    "Feature Importance"
])

# ============================================================
# TAB 1 — THREAT ANALYSIS
# ============================================================
with tab1:
    st.header("Threat Analysis Dashboard")

    if 'model' not in st.session_state:
        st.info("Waiting for model training. Click 'Train Model Now' in the sidebar.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Packet Simulation")
            st.info("Pick a random packet from the test data to simulate live traffic.")

            if st.button("Capture Random Packet"):
                random_idx = np.random.randint(0, len(st.session_state['X_test']))
                packet_data = st.session_state['X_test'].iloc[random_idx]
                actual_label = st.session_state['y_test'].iloc[random_idx]

                st.session_state['current_packet'] = packet_data
                st.session_state['actual_label'] = actual_label

        if 'current_packet' in st.session_state:
            packet = st.session_state['current_packet']

            with col1:
                st.write("*Packet Header Info:*")
                st.dataframe(packet, use_container_width=True)

            with col2:
                st.subheader("AI Detection Result")
                prediction = st.session_state['model'].predict([packet])[0]

                if prediction == "BENIGN":
                    st.success("STATUS: SAFE (BENIGN)")
                else:
                    st.error(f"STATUS: ATTACK DETECTED ({prediction})")

                st.caption(f"Ground Truth Label: {st.session_state['actual_label']}")

                correct = prediction == st.session_state['actual_label']
                if correct:
                    st.success("Prediction matches ground truth")
                else:
                    st.warning("Prediction does NOT match ground truth")

                st.markdown("---")
                st.subheader("Ask AI Analyst (Groq)")

                if st.button("Generate Explanation"):
                    if not groq_api_key:
                        st.warning("add groq api key")
                    else:
                        try:
                            client = Groq(api_key=groq_api_key)

                            prompt = f"""
                            You are a cybersecurity analyst.
                            A network packet was detected as: {prediction}.

                            Packet Technical Details:
                            {packet.to_string()}

                            Please explain:
                            1. Why these specific values (like Flow Duration or Packet Length) might indicate {prediction}.
                            2. If it is BENIGN, explain why it looks normal.
                            3. Keep the answer short and simple for a student.
                            """

                            with st.spinner("Groq is analyzing the packet..."):
                                completion = client.chat.completions.create(
                                    model="llama-3.3-70b-versatile",
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=0.6,
                                )
                                st.info(completion.choices[0].message.content)

                        except Exception as e:
                            st.error(f"API Error: {e}")

# ============================================================
# TAB 2 — MODEL EVALUATION
# ============================================================
with tab2:
    st.header("Model Evaluation")

    if 'model' not in st.session_state:
        st.info("Train the model first to see evaluation metrics.")
    else:
        y_pred = st.session_state['model'].predict(st.session_state['X_test'])
        y_test = st.session_state['y_test']

        # --- METRICS ROW ---
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.2%}")

        labels = [l for l in report if l not in ['accuracy', 'macro avg', 'weighted avg']]
        attack_label = [l for l in labels if l != 'BENIGN']

        if attack_label:
            attack = attack_label[0]
            col2.metric("Precision", f"{report[attack]['precision']:.2%}")
            col3.metric("Recall", f"{report[attack]['recall']:.2%}")
            col4.metric("F1 Score", f"{report[attack]['f1-score']:.2%}")

        st.markdown("---")

        col_a, col_b = st.columns(2)

        # --- CONFUSION MATRIX ---
        with col_a:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=st.session_state['model'].classes_,
                yticklabels=st.session_state['model'].classes_,
                ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

        # --- CLASSIFICATION REPORT TABLE ---
        with col_b:
            st.subheader("Classification Report")
            report_df = pd.DataFrame(report).transpose()
            report_df = report_df.round(3)
            st.dataframe(report_df, use_container_width=True)

            st.markdown("---")
            st.subheader("Class Distribution in Test Set")
            class_counts = y_test.value_counts()
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            class_counts.plot(kind='bar', ax=ax2, color=['#2ecc71', '#e74c3c'])
            ax2.set_title("BENIGN vs ATTACK")
            ax2.set_xlabel("Class")
            ax2.set_ylabel("Count")
            plt.xticks(rotation=0)
            st.pyplot(fig2)

# ============================================================
# TAB 3 — FEATURE IMPORTANCE
# ============================================================
with tab3:
    st.header("Feature Importance")

    if 'model' not in st.session_state:
        st.info("Train the model first to see feature importance.")
    else:
        importances = st.session_state['model'].feature_importances_
        feature_names = st.session_state['features']

        feat_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Importance Chart")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(
                data=feat_df,
                x='Importance',
                y='Feature',
                palette='viridis',
                ax=ax
            )
            ax.set_title("Feature Importance (Random Forest)")
            ax.set_xlabel("Importance Score")
            st.pyplot(fig)

        with col2:
            st.subheader("Importance Table")
            feat_df['Importance %'] = (feat_df['Importance'] * 100).round(2)
            st.dataframe(feat_df, use_container_width=True)

            st.markdown("---")
            st.subheader("What each feature means")
            st.markdown("""
            - *Flow Duration* — total time of the network flow in microseconds
            - *Total Fwd Packets* — packets sent from source to destination
            - *Total Backward Packets* — packets sent back from destination
            - *Total Length of Fwd Packets* — total size of forward packets
            - *Fwd Packet Length Max* — largest forward packet size
            - *Flow IAT Mean* — average time between packets in the flow
            - *Flow IAT Std* — variation in time between packets
            - *Flow Packets/s* — how many packets per second in the flow
            """)

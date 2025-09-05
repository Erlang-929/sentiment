import streamlit as st
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer, AutoTokenizer
import pandas as pd
import plotly.express as px
import numpy as np
from huggingface_hub import hf_hub_download
import re
import emoji
import time

with st.sidebar:
    st.title("About this App 🤔")
    st.write("This app can classify emotions from Indonesian text into five categories: **SADNESS**, **ANGER**, **SUPPORT**, **HOPE**, and **DISAPPOINTMENT**.")
    st.write("It uses Natural Language Processing (NLP) to automatically analyze and understand human language, enabling tasks such as sentiment analysis, text classification, and emotion detection.")
    st.write("The model is based on IndoBERT, a deep learning model pre-trained on a large collection of Indonesian texts from sources like Wikipedia, news articles, and online blogs etc. It is then fine-tuned for emotion classification using a dataset of Indonesian tweets and Instagram comments.")

st.title("Text Emotion Classification")
st.write("**NOTE** This app is designed for Indonesian text classification only.")

tab1, tab2 = st.tabs(["Classify Single Text", "Classify Dataset"])

model_path = hf_hub_download(
    repo_id="erlangram/text_emotion_model", 
    filename="model _indobert2.h5")

model = tf.keras.models.load_model(
    model_path,
    compile=False,
    custom_objects={"TFBertModel": TFBertModel}
)

tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
label_encode = {0: 'SADNESS', 1: 'ANGER', 2: 'SUPPORT', 3: 'HOPE', 4: 'DISAPPOINTMENT'}

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[-+]?[0-9]+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = emoji.replace_emoji(text, replace='')  # hapus emoji
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

emoji_label = {
    "SADNESS": "😢",
    "ANGER": "😡",
    "SUPPORT": "💪",
    "HOPE": "🌟",
    "DISAPPOINTMENT": "😞"
}


with tab1:
    text = st.text_area("Input Text:", "")

    if st.button("Classify 🔍️"):
        if text.strip() == "":
            st.warning("Please enter some text.")
        else:
            # Preprocess
            clean_text = preprocess(text)

            # Tokenize
            encoded = tokenizer(
                clean_text,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=100,
                return_tensors='tf'
            )

            # Predict
            pred = model.predict(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"]
                }
            )

            probs = pred[0]  # ambil probabilitas tiap label
            pred_label = np.argmax(probs)
            prediction = label_encode[pred_label]

            col1, col2 = st.columns(2)

            with col1:
                st.success("**Input Text**")
                st.write(text)

                st.success("**Prediction Probabilities**")
                st.table({
                    "Label": [label_encode[i] for i in range(len(probs))],
                    "Probability": probs
                })

           
            with col2:
                pred_label = np.argmax(pred, axis=1)[0]

                st.success("**Result**")
                st.write(f"Emotion Detected: **{prediction}** {emoji_label.get(prediction)}")

                df = pd.DataFrame({
                "Emotion": [label_encode[i] for i in range(len(probs))],
                "Probability": probs
                })

                # Plot multicolor bar chart
                fig = px.bar(
                    df,
                    x="Emotion",
                    y="Probability",
                    color="Emotion",
                    text="Probability",
                    title="Probability Distribution of Emotions",
                    range_y=[0, 1]
                )
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.subheader("Classify Emotion from Dataset")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)

        if "text" not in df_input.columns:
            st.error("File must contain 'text' column.")
        else:
            if st.button("Classify Dataset"):
                results = []

                n = len(df_input)  # total baris
                progress_text = "Classifying dataset..."
                my_bar = st.progress(0, text=progress_text)

                for i, t in enumerate(df_input["text"]):
                    clean_t = preprocess(str(t))
                    encoded = tokenizer(
                        clean_t,
                        add_special_tokens=True,
                        padding='max_length',
                        truncation=True,
                        max_length=100,
                        return_tensors='tf'
                    )
                    pred = model.predict(
                        {
                            "input_ids": encoded["input_ids"],
                            "attention_mask": encoded["attention_mask"]
                        }
                    )
                    probs = pred[0]
                    label = label_encode[np.argmax(probs)]
                    results.append(label)

                    my_bar.progress((i+1)/n, text=f"Processing row {i+1}/{n}")

                my_bar.empty()
                df_input["label"] = results
                st.success("Done!")
                st.dataframe(df_input)

                # Download hasil prediksi
                csv = df_input.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )

                label_counts = df_input['label'].value_counts().reset_index()
                label_counts.columns = ['Emotion', 'Count']

                # Pie chart
                fig = px.pie(
                    label_counts,
                    names='Emotion',
                    values='Count',
                    color='Emotion',
                    color_discrete_map={
                        "SADNESS": "blue",
                        "ANGER": "red",
                        "SUPPORT": "green",
                        "HOPE": "yellow",
                        "DISAPPOINTMENT": "purple"
                    },
                    #title="Overall Emotion Distribution from Dataset"
                )
                fig.update_traces(textinfo='percent+label')

                st.success("**Overall Emotion Distribution from Dataset**")
                st.plotly_chart(fig, use_container_width=True)

                st.success("**Emotion Counts**")
                st.table(label_counts)


import requests
import streamlit as st
import pandas as pd

DEBUG = True
PROD_URL = "https://ml-streamlit-app.onrender.com"

st.title("📊 Visualisation app")

st.write(
    "We are so glad to see you here. ✨ "
)

def start_prediction():
    data = requests.get("https://ml-streamlit-app.onrender.com/start/")
    print(data.json())
    return data.json()


def fetch_anomalies():
    data = requests.get("https://ml-streamlit-app.onrender.com/anomalies/")
    print(data.json())
    data = data.json()
    

    df = pd.DataFrame(data)

    st.dataframe(df, use_container_width=True)

if st.button("Start prediction...", on_click=start_prediction):
    fetch_anomalies()

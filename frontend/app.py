import requests
import streamlit as st
import pandas as pd

st.title("📊 Visualisation app")

st.write(
    "We are so glad to see you here. ✨ "
)

def start_prediction():
    data = requests.get("http://localhost:8000/start/")
    print(data.json())
    return data.json()


def fetch_anomalies():
    data = requests.get("http://localhost:8000/anomalies/")
    print(data.json())
    data = data.json()
    

    df = pd.DataFrame(data)

    st.dataframe(df, use_container_width=True)

if st.button("Start prediction...", on_click=start_prediction):
    fetch_anomalies()

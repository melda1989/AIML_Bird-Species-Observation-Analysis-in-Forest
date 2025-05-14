import streamlit as st
import pandas as pd
import plotly.express as px
df = pd.read_csv("C:\\Users\\DAVID\\Downloads\\cleaned_combined_data.csv")
st.title("Bird Observation Dashboard")
# Filters
habitat = st.selectbox("Select Habitat Type", df["Habitat_Type"].unique())
df_filtered = df[df["Habitat_Type"] == habitat]
# Visuals
fig = px.histogram(df_filtered, x="Scientific_Name", title="Species Frequency")
st.plotly_chart(fig)
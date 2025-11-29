# ----------------------------------------------------------
# app.py — Streamlit Dashboard for Weather Data
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Weather Dashboard", layout="wide")

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

@st.cache_data
def load_data():
    countries_weather_df = pd.read_csv("countries.csv")
    cities_weather_df = pd.read_csv("cities.csv")
    return countries_weather_df, cities_weather_df

countries_weather_df, cities_weather_df = load_data()

countries_weather_df.drop_duplicates(inplace=True)
countries_weather_df.dropna( inplace=True)
cities_weather_df.drop_duplicates(inplace=True)
cities_weather_df.dropna(inplace=True)

# ----------------------------------------------------------
# PREPARE COLUMNS
# ----------------------------------------------------------

num_cols_countries = countries_weather_df.select_dtypes(include="number").columns
num_cols_cities = cities_weather_df.select_dtypes(include="number").columns

cat_cols_countries = countries_weather_df.select_dtypes(exclude="number").columns
cat_cols_cities = cities_weather_df.select_dtypes(exclude="number").columns


# ----------------------------------------------------------
# NORMALIZATION
# ----------------------------------------------------------

scaler = StandardScaler()
countries_norm = countries_weather_df.copy()
cities_norm = cities_weather_df.copy()

countries_norm[num_cols_countries] = scaler.fit_transform(countries_weather_df[num_cols_countries])
cities_norm[num_cols_cities] = scaler.fit_transform(cities_weather_df[num_cols_cities])


# ----------------------------------------------------------
# STREAMLIT MULTIPAGE NAVIGATION
# ----------------------------------------------------------

page = st.sidebar.radio(
    "Навигация",
    ["📊 Исходные данные", "🧠 Анализ данных"]
)

st.sidebar.info("Weather Dashboard — Streamlit")


# ==========================================================
# PAGE 1 — RAW DATA VISUALIZATION
# ==========================================================

if page == "📊 Исходные данные":

    st.title("📊 Визуализация исходных данных")

    st.header("Датасеты")
    tab1, tab2 = st.tabs(["🌍 Countries", "🏙 Cities"])

    # TABLES -----------------------------------------------------
    with tab1:
        st.subheader("Таблица данных — Countries")
        st.dataframe(countries_weather_df)

        st.subheader("Базовые статистики")
        st.write(countries_weather_df.describe(include="all"))

    with tab2:
        st.subheader("Таблица данных — Cities")
        st.dataframe(cities_weather_df)

        st.subheader("Базовые статистики")
        st.write(cities_weather_df.describe(include="all"))

    # VISUALIZATION CONTROLS -------------------------------------
    st.header("Графики")

    df_choice = st.selectbox("Выберите датасет:", ["Countries", "Cities"])
    if df_choice == "Countries":
        df_raw = countries_weather_df
        num_cols = num_cols_countries
        cat_cols = cat_cols_countries
    else:
        df_raw = cities_weather_df
        num_cols = num_cols_cities
        cat_cols = cat_cols_cities

    graph_type = st.selectbox(
        "Выберите тип графика",
        ["Histogram", "Boxplot", "Scatter plot", "Category Bar", "Correlation Heatmap"]
    )

    # HISTOGRAM ---------------------------------------------------
    if graph_type == "Histogram":
        col = st.selectbox("Числовой признак:", num_cols)
        fig = px.histogram(df_raw, x=col, nbins=30)
        st.plotly_chart(fig, use_container_width=True)

    # BOXPLOT ------------------------------------------------------
    if graph_type == "Boxplot":
        col = st.selectbox("Числовой признак:", num_cols)
    
        fig = px.box(
            df_raw,
            x=pd.Series(["value"] * len(df_raw)),
            y=col,
            labels={"x": "", col: col},
            title=f"Boxplot for {col}"
        )

    st.plotly_chart(fig, use_container_width=True)
    # SCATTER ------------------------------------------------------
    if graph_type == "Scatter plot":
        x = st.selectbox("X:", num_cols)
        y = st.selectbox("Y:", num_cols)
        fig = px.scatter(df_raw, x=x, y=y)
        st.plotly_chart(fig, use_container_width=True)

    # CATEGORY BAR -------------------------------------------------
    if graph_type == "Category Bar":
        col = st.selectbox("Категориальный признак:", cat_cols)
    
        # Подсчет категорий
        counts = df_raw[col].value_counts().reset_index()
    
        # Переименуем колонки
        counts.columns = [col, "count"]
    
        # Корректный bar chart
        fig = px.bar(
            counts,
            x=col,
            y="count",
            title=f"Распределение категорий: {col}"
        )
    
        st.plotly_chart(fig, use_container_width=True)

    # CORRELATION HEATMAP ------------------------------------------
    if graph_type == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_raw[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)


# ==========================================================
# PAGE 2 — ANALYSIS RESULTS (KMeans + Regression)
# ==========================================================

if page == "🧠 Анализ данных":

    st.title("🧠 Результаты анализа данных")

    analysis_type = st.selectbox(
        "Выберите метод анализа:",
        ["Clustering (K-Means)", "Linear Regression"]
    )

    df_choice = st.selectbox("Выберите датасет:", ["Countries", "Cities"])
    if df_choice == "Countries":
        df_norm = countries_norm
        df_raw = countries_weather_df
        num_cols = num_cols_countries
    else:
        df_norm = cities_norm
        df_raw = cities_weather_df
        num_cols = num_cols_cities


    # --------------------------------------------------------------
    # K-MEANS CLUSTERING
    # --------------------------------------------------------------
    if analysis_type == "Clustering (K-Means)":

        st.header("Кластеризация K-Means")

        k = st.slider("Количество кластеров", 2, 10, 3)
        x = st.selectbox("X признак", num_cols)
        y = st.selectbox("Y признак", num_cols)

        model = KMeans(n_clusters=k)
        clusters = model.fit_predict(df_norm[[x, y]])

        df_raw["Cluster"] = clusters

        fig = px.scatter(
            df_raw,
            x=x,
            y=y,
            color="Cluster",
            title="Кластеры K-Means",
            hover_data=df_raw.columns
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Информация о кластерах")
        st.write(df_raw.groupby("Cluster")[num_cols].mean())


    # --------------------------------------------------------------
    # LINEAR REGRESSION
    # --------------------------------------------------------------
    if analysis_type == "Linear Regression":

        st.header("Линейная регрессия")

        x = st.selectbox("Признак X", num_cols)
        y = st.selectbox("Признак Y (цель)", num_cols)

        model = LinearRegression()
        model.fit(df_norm[[x]], df_norm[y])

        pred = model.predict(df_norm[[x]])

        fig = px.scatter(df_raw, x=x, y=y, title="Линейная регрессия")
        fig.add_traces(px.line(df_raw, x=df_raw[x], y=pred).data)

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Коэффициенты модели")
        st.write({
            "intercept": model.intercept_,
            "coef": model.coef_[0]
        })


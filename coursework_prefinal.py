# ----------------------------------------------------------
# app.py — Оптимизированный Streamlit Dashboard для Weather Data
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ML библиотеки
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Weather Analytics Dashboard", 
    layout="wide"
)

# ----------------------------------------------------------
# LOAD AND CACHE DATA
# ----------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    """Загрузка и предобработка данных с кэшированием"""
    try:
        countries_df = pd.read_csv("countries.csv")
    except:
        countries_df = pd.DataFrame()
    
    try:
        cities_df = pd.read_csv("cities.csv")
    except:
        cities_df = pd.DataFrame()
    
    try:
        daily_df = pd.read_csv("daily_weather_smallest.csv")
        if 'date' in daily_df.columns:
            daily_df['date'] = pd.to_datetime(daily_df['date'], errors='coerce')
            # Добавляем только базовые временные признаки для оптимизации
            daily_df['month'] = daily_df['date'].dt.month
    except:
        daily_df = pd.DataFrame()
    
    return countries_df, cities_df, daily_df

# Загрузка данных
countries_df, cities_df, daily_df = load_data()

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
st.sidebar.title("Weather Analytics")

# Навигация
page = st.sidebar.radio(
    "Навигация",
    ["Визуализация данных", "Анализ данных"]
)

# Глобальные фильтры
if not daily_df.empty and 'date' in daily_df.columns:
    st.sidebar.header("Фильтры")
    
    try:
        min_date = daily_df['date'].min()
        max_date = daily_df['date'].max()
        
        date_range = st.sidebar.date_input(
            "Диапазон дат:",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
    except:
        pass

# ==========================================================
# PAGE 1 — ВИЗУАЛИЗАЦИЯ ДАННЫХ
# ==========================================================
if page == "Визуализация данных":
    
    st.title("Визуализация погодных данных")
    
    # Быстрые KPI
    if not daily_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'city_name' in daily_df.columns:
                st.metric("Городов", daily_df['city_name'].nunique())
            else:
                st.metric("Записей", len(daily_df))
        
        with col2:
            if 'date' in daily_df.columns:
                st.metric("Период", f"{len(daily_df['date'].dt.date.unique())} дней")
        
        with col3:
            numeric_cols = daily_df.select_dtypes(include=[np.number]).columns
            st.metric("Признаков", len(numeric_cols))
        
        with col4:
            missing = daily_df.isnull().sum().sum()
            st.metric("Пропусков", missing)
    
    # Вкладки
    tab1, tab2, tab3 = st.tabs(["Данные", "Графики", "Карта"])
    
    with tab1:
        st.header("Просмотр данных")
        
        dataset_choice = st.selectbox(
            "Выберите датасет:",
            ["Ежедневные данные", "Города", "Страны"]
        )
        
        if dataset_choice == "Ежедневные данные" and not daily_df.empty:
            df_display = daily_df
            
            # Быстрый фильтр по городу
            if 'city_name' in daily_df.columns:
                cities = daily_df['city_name'].unique().tolist()
                selected_city = st.selectbox("Фильтр по городу:", ['Все'] + cities[:20])
                if selected_city != 'Все':
                    df_display = df_display[df_display['city_name'] == selected_city]
            
            # Быстрый фильтр по дате
            if 'date' in df_display.columns:
                dates = df_display['date'].dt.date.unique()
                if len(dates) > 0:
                    min_date, max_date = dates.min(), dates.max()
                    selected_date = st.date_input(
                        "Фильтр по дате:",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date
                    )
                    df_display = df_display[df_display['date'].dt.date == selected_date]
        
        elif dataset_choice == "Города" and not cities_df.empty:
            df_display = cities_df
        elif dataset_choice == "Страны" and not countries_df.empty:
            df_display = countries_df
        else:
            st.warning("Данный датасет не загружен")
            df_display = pd.DataFrame()
        
        if not df_display.empty:
            # Отображение таблицы с пагинацией
            rows_per_page = st.selectbox("Строк на странице:", [50, 100, 200], index=0)
            page_num = st.number_input("Страница:", min_value=1, value=1)
            
            start_idx = (page_num - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            
            st.dataframe(
                df_display.iloc[start_idx:end_idx],
                width='stretch',
                height=400
            )
            
            # Быстрая статистика
            if st.checkbox("Показать статистику", value=False):
                numeric_cols = df_display.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.dataframe(df_display[numeric_cols].describe(), width='stretch')
    
    with tab2:
        st.header("Визуализация данных")
        
        if not daily_df.empty:
            # Быстрый выбор типа графика и данных
            col1, col2 = st.columns(2)
            
            with col1:
                chart_type = st.selectbox(
                    "Тип графика:",
                    ["Гистограмма", "Box Plot", "Scatter Plot", "Линейный график"]
                )
            
            with col2:
                numeric_cols = daily_df.select_dtypes(include=[np.number]).columns.tolist()
                if chart_type == "Scatter Plot":
                    x_col = st.selectbox("X:", numeric_cols[:5])
                    y_col = st.selectbox("Y:", numeric_cols[:5])
                else:
                    col_selected = st.selectbox("Колонка:", numeric_cols[:10])
            
            # Быстрое создание графиков
            if chart_type == "Гистограмма":
                fig = px.histogram(daily_df, x=col_selected, nbins=30)
                st.plotly_chart(fig, width='stretch', use_container_width=True)
            
            elif chart_type == "Box Plot":
                fig = px.box(daily_df, y=col_selected)
                st.plotly_chart(fig, width='stretch', use_container_width=True)
            
            elif chart_type == "Scatter Plot":
                fig = px.scatter(daily_df, x=x_col, y=y_col, opacity=0.5)
                st.plotly_chart(fig, width='stretch', use_container_width=True)
            
            elif chart_type == "Линейный график" and 'date' in daily_df.columns:
                # Агрегация для скорости
                sample_df = daily_df.copy()
                if len(sample_df) > 10000:
                    sample_df = sample_df.sample(10000)
                
                # Группировка по дате
                agg_df = sample_df.groupby('date')[col_selected].mean().reset_index()
                fig = px.line(agg_df, x='date', y=col_selected)
                st.plotly_chart(fig, width='stretch', use_container_width=True)
            
            # Быстрая корреляционная матрица
            if st.checkbox("Показать корреляционную матрицу", value=False):
                if len(numeric_cols) > 1:
                    # Используем только числовые колонки
                    corr_data = daily_df[numeric_cols].corr()
                    
                    # Ограничиваем размер для скорости
                    if len(corr_data) > 10:
                        corr_data = corr_data.iloc[:10, :10]
                    
                    fig = px.imshow(corr_data, text_auto=True, aspect="auto")
                    st.plotly_chart(fig, width='stretch', use_container_width=True)
    
    with tab3:
        st.header("Географическая визуализация")
        
        if not cities_df.empty and 'latitude' in cities_df.columns and 'longitude' in cities_df.columns:
            # Простая карта без сложных опций
            fig = px.scatter_geo(
                cities_df,
                lat='latitude',
                lon='longitude',
                hover_name='city_name' if 'city_name' in cities_df.columns else None,
                title='Распределение городов'
            )
            st.plotly_chart(fig, width='stretch', use_container_width=True)

# ==========================================================
# PAGE 2 — АНАЛИЗ ДАННЫХ
# ==========================================================
else:
    
    st.title("Анализ погодных данных")
    
    # Используем только daily_df для анализа
    if daily_df.empty:
        st.error("Для анализа нужны ежедневные данные")
    else:
        # Быстрая предобработка
        df_analysis = daily_df.copy()
        numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.error("Нет числовых колонок для анализа")
        else:
            # Выбор метода анализа
            analysis_method = st.selectbox(
                "Метод анализа:",
                ["Кластеризация", "Регрессия", "PCA"]
            )
            
            # Быстрая стандартизация
            scaler = StandardScaler()
            df_numeric = df_analysis[numeric_cols].fillna(0)
            if len(df_numeric) > 0:
                df_scaled_numeric = scaler.fit_transform(df_numeric)
                df_scaled = pd.DataFrame(df_scaled_numeric, columns=numeric_cols)
            else:
                df_scaled = pd.DataFrame()
            
            # ========== КЛАСТЕРИЗАЦИЯ ==========
            if analysis_method == "Кластеризация":
                st.header("Кластеризация K-Means")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Ограничиваем выбор признаков для скорости
                    features = st.multiselect(
                        "Признаки:",
                        numeric_cols[:8],  # Только первые 8 для скорости
                        default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
                    )
                
                with col2:
                    n_clusters = st.slider("Кластеров:", 2, 8, 3)
                
                if len(features) >= 2:
                    # Ограничиваем данные для скорости
                    X = df_scaled[features]
                    if len(X) > 5000:
                        X_sample = X.sample(5000, random_state=42)
                    else:
                        X_sample = X
                    
                    # Быстрая кластеризация
                    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                    clusters = kmeans.fit_predict(X_sample)
                    
                    # Визуализация
                    df_viz = df_analysis.loc[X_sample.index].copy()
                    df_viz['Cluster'] = clusters
                    
                    fig = px.scatter(
                        df_viz,
                        x=features[0],
                        y=features[1],
                        color='Cluster',
                        title=f"Кластеризация: {features[0]} vs {features[1]}"
                    )
                    
                    st.plotly_chart(fig, width='stretch', use_container_width=True)
                    
                    # Быстрая статистика кластеров
                    if 'Cluster' in df_viz.columns:
                        cluster_stats = df_viz.groupby('Cluster')[numeric_cols[:5]].mean()
                        st.dataframe(cluster_stats, width='stretch')
            
            # ========== РЕГРЕССИЯ ==========
            elif analysis_method == "Регрессия":
                st.header("Линейная регрессия")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    target = st.selectbox("Целевая (Y):", numeric_cols[:5])
                
                with col2:
                    available = [col for col in numeric_cols[:5] if col != target]
                    feature = st.selectbox("Признак (X):", available)
                
                if target and feature:
                    # Простая регрессия с одним признаком
                    X = df_scaled[[feature]].values.reshape(-1, 1)
                    y = df_scaled[target].values
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    
                    # Метрики
                    r2 = r2_score(y, y_pred)
                    mae = mean_absolute_error(y, y_pred)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R²", f"{r2:.3f}")
                    with col2:
                        st.metric("MAE", f"{mae:.3f}")
                    
                    # Простая визуализация
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=df_analysis[feature],
                        y=df_analysis[target],
                        mode='markers',
                        name='Данные',
                        marker=dict(opacity=0.3)
                    ))
                    
                    # Линия регрессии
                    x_range = np.linspace(df_analysis[feature].min(), df_analysis[feature].max(), 100)
                    x_range_scaled = scaler.transform(x_range.reshape(-1, 1))
                    y_pred_range = model.predict(x_range_scaled)
                    
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_pred_range,
                        mode='lines',
                        name='Регрессия',
                        line=dict(color='red', width=3)
                    ))
                    
                    fig.update_layout(
                        title=f"Регрессия: {target} от {feature}",
                        xaxis_title=feature,
                        yaxis_title=target
                    )
                    
                    st.plotly_chart(fig, width='stretch', use_container_width=True)
            
            # ========== PCA ==========
            else:
                st.header("Анализ главных компонент (PCA)")
                
                # Ограничиваем количество признаков для скорости
                pca_features = st.multiselect(
                    "Признаки для PCA:",
                    numeric_cols[:8],
                    default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
                )
                
                if len(pca_features) >= 2:
                    n_components = st.slider("Компонент:", 2, 4, 2)
                    
                    # Ограничиваем данные для скорости
                    X_pca = df_scaled[pca_features]
                    if len(X_pca) > 5000:
                        X_pca_sample = X_pca.sample(5000, random_state=42)
                    else:
                        X_pca_sample = X_pca
                    
                    # Быстрый PCA
                    pca = PCA(n_components=n_components)
                    X_pca_transformed = pca.fit_transform(X_pca_sample)
                    
                    # Объясненная дисперсия
                    explained_variance = pca.explained_variance_ratio_
                    
                    # Простой график дисперсии
                    fig_var = go.Figure()
                    
                    fig_var.add_trace(go.Bar(
                        x=[f"PC{i+1}" for i in range(n_components)],
                        y=explained_variance,
                        name='Доля дисперсии'
                    ))
                    
                    fig_var.update_layout(
                        title="Объясненная дисперсия",
                        yaxis_title='Доля дисперсии'
                    )
                    
                    st.plotly_chart(fig_var, width='stretch', use_container_width=True)
                    
                    # 2D визуализация PCA
                    if n_components >= 2:
                        df_pca_viz = df_analysis.loc[X_pca_sample.index].copy()
                        df_pca_viz['PC1'] = X_pca_transformed[:, 0]
                        df_pca_viz['PC2'] = X_pca_transformed[:, 1]
                        
                        fig_pca = px.scatter(
                            df_pca_viz,
                            x='PC1',
                            y='PC2',
                            title="PCA - Первые две компоненты"
                        )
                        
                        st.plotly_chart(fig_pca, width='stretch', use_container_width=True)

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.caption("Weather Analytics Dashboard")

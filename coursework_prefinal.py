# ----------------------------------------------------------
# app.py — Streamlit Dashboard for Weather Data (Simplified)
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Weather Analytics Dashboard", 
    layout="wide",
    page_icon="🌤️"
)

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
@st.cache_data
def load_data():
    # Проверяем наличие файлов с несколькими расширениями
    import os
    
    # Список возможных имен файлов
    files_to_try = {
        'countries': ['countries.csv', 'countries_weather.csv', 'countries_data.csv'],
        'cities': ['cities.csv', 'cities_weather.csv', 'cities_data.csv'],
        'daily': ['daily_weather_smallest.csv', 'daily_weather.csv', 'daily.csv']
    }
    
    dataframes = {}
    
    for name, filenames in files_to_try.items():
        df = None
        for filename in filenames:
            try:
                if os.path.exists(filename):
                    df = pd.read_csv(filename)
                    st.sidebar.success(f"✓ Загружен {filename}")
                    break
            except:
                continue
        
        if df is None:
            # Создаем пустой датафрейм если файл не найден
            df = pd.DataFrame()
            st.sidebar.warning(f"⚠ Файл {name} не найден")
        
        dataframes[name] = df
    
    return dataframes['countries'], dataframes['cities'], dataframes['daily']

countries_weather_df, cities_weather_df, daily_weather_df = load_data()

# ----------------------------------------------------------
# DATA PREPROCESSING
# ----------------------------------------------------------
def preprocess_dataframes():
    """Предобработка всех датафреймов"""
    processed_dfs = []
    
    for df, name in zip([countries_weather_df, cities_weather_df, daily_weather_df], 
                        ['countries', 'cities', 'daily']):
        if df.empty:
            processed_dfs.append(df)
            continue
            
        df_clean = df.copy()
        
        # Удаление дубликатов
        df_clean.drop_duplicates(inplace=True)
        
        # Для daily weather добавляем временные признаки
        if name == 'daily' and not df_clean.empty:
            if 'date' in df_clean.columns:
                try:
                    df_clean['date'] = pd.to_datetime(df_clean['date'])
                    df_clean['year'] = df_clean['date'].dt.year
                    df_clean['month'] = df_clean['date'].dt.month
                    df_clean['day'] = df_clean['date'].dt.day
                    
                    # Сезоны
                    df_clean['season'] = df_clean['month'] % 12 // 3 + 1
                    season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
                    df_clean['season_name'] = df_clean['season'].map(season_map)
                except:
                    pass
        
        processed_dfs.append(df_clean)
    
    return processed_dfs

countries_weather_df, cities_weather_df, daily_weather_df = preprocess_dataframes()

# ----------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------
def calculate_autocorrelation(series, max_lags=50):
    """Вычисляет автокорреляцию без statsmodels"""
    if len(series) < 2:
        return []
    
    series_clean = series.dropna()
    if len(series_clean) < 2:
        return []
    
    autocorr = []
    n = len(series_clean)
    mean = series_clean.mean()
    var = series_clean.var()
    
    if var == 0:
        return [0] * min(max_lags, n-1)
    
    max_lags = min(max_lags, n-1)
    
    for lag in range(1, max_lags + 1):
        if lag < n:
            numerator = ((series_clean - mean) * (series_clean.shift(lag) - mean)).sum()
            denominator = (n - lag) * var
            autocorr.append(numerator / denominator if denominator != 0 else 0)
        else:
            autocorr.append(0)
    
    return autocorr

def decompose_time_series(df, value_col, trend_window=30, seasonal_period=365):
    """Простая декомпозиция временного ряда"""
    if df.empty or value_col not in df.columns:
        return None
    
    result = df.copy()
    
    # Тренд - скользящее среднее
    result['trend'] = result[value_col].rolling(
        window=min(trend_window, len(result)), 
        center=True, 
        min_periods=1
    ).mean()
    
    # Сезонность (упрощенная)
    if seasonal_period < len(result):
        # Группируем по позиции в периоде
        result['position'] = result.index % seasonal_period
        seasonal_pattern = result.groupby('position')[value_col].mean()
        
        # Сопоставляем паттерн с данными
        result['seasonal'] = result['position'].map(seasonal_pattern)
        result['seasonal'].fillna(result[value_col].mean(), inplace=True)
    else:
        result['seasonal'] = 0
    
    # Остаток
    result['residual'] = result[value_col] - result['trend'] - result['seasonal']
    
    return result

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
st.sidebar.title("🌤️ Weather Analytics")
st.sidebar.markdown("---")

# Показать информацию о загруженных данных
st.sidebar.subheader("Загруженные данные")
if not countries_weather_df.empty:
    st.sidebar.info(f"🌍 Countries: {len(countries_weather_df)} записей")
if not cities_weather_df.empty:
    st.sidebar.info(f"🏙️ Cities: {len(cities_weather_df)} записей")
if not daily_weather_df.empty:
    st.sidebar.info(f"📅 Daily: {len(daily_weather_df)} записей")

# Навигация
page = st.sidebar.radio(
    "Навигация",
    ["📊 Визуализация данных", "🔍 Анализ", "📈 Прогнозирование", "ℹ️ О проекте"]
)

st.sidebar.markdown("---")

# Глобальные фильтры
if not daily_weather_df.empty and 'date' in daily_weather_df.columns:
    st.sidebar.header("Фильтры")
    
    # Фильтр по дате
    min_date = daily_weather_df['date'].min()
    max_date = daily_weather_df['date'].max()
    
    if isinstance(min_date, str):
        min_date = pd.to_datetime(min_date)
    if isinstance(max_date, str):
        max_date = pd.to_datetime(max_date)
    
    date_range = st.sidebar.date_input(
        "Диапазон дат:",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

st.sidebar.markdown("---")
st.sidebar.info("""
**Инструкция:**
1. Выберите страницу навигации
2. Используйте фильтры для настройки
3. Взаимодействуйте с графиками
""")

# ==========================================================
# PAGE 1 — RAW DATA VISUALIZATION
# ==========================================================
if page == "📊 Визуализация данных":
    
    st.title("📊 Визуализация погодных данных")
    
    # Проверка наличия данных
    if daily_weather_df.empty and cities_weather_df.empty and countries_weather_df.empty:
        st.error("❌ Данные не загружены. Убедитесь, что CSV файлы находятся в корневой папке.")
        st.info("""
        Необходимые файлы:
        - `countries.csv` или `countries_weather.csv`
        - `cities.csv` или `cities_weather.csv`
        - `daily_weather_smallest.csv` или `daily_weather.csv`
        """)
        
        # Создание демо данных для тестирования
        if st.button("Создать демо-данные для тестирования"):
            # Демо данные для countries
            demo_countries = pd.DataFrame({
                'country': ['USA', 'Canada', 'UK', 'Germany', 'France'],
                'avg_temp': [15.5, 5.2, 10.1, 9.8, 12.3],
                'avg_precipitation': [800, 900, 1100, 700, 750],
                'elevation': [500, 1000, 200, 300, 400]
            })
            
            # Демо данные для cities
            demo_cities = pd.DataFrame({
                'city_name': ['New York', 'Toronto', 'London', 'Berlin', 'Paris'],
                'country': ['USA', 'Canada', 'UK', 'Germany', 'France'],
                'latitude': [40.7128, 43.6532, 51.5074, 52.5200, 48.8566],
                'longitude': [-74.0060, -79.3832, -0.1278, 13.4050, 2.3522],
                'population': [8419000, 2930000, 8982000, 3769000, 2148000]
            })
            
            # Демо данные для daily weather
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            demo_daily = pd.DataFrame({
                'date': dates,
                'city_name': np.random.choice(['New York', 'Toronto', 'London'], len(dates)),
                'temperature': np.random.normal(15, 5, len(dates)),
                'precipitation': np.random.exponential(2, len(dates)),
                'humidity': np.random.uniform(40, 90, len(dates)),
                'wind_speed': np.random.exponential(5, len(dates))
            })
            
            countries_weather_df = demo_countries
            cities_weather_df = demo_cities
            daily_weather_df = demo_daily
            
            st.success("✅ Демо-данные созданы! Продолжайте работу с дашбордом.")
            st.rerun()
    
    else:
        # KPI Cards
        st.subheader("📈 Ключевые показатели")
        
        # Создаем колонки для KPI
        if not daily_weather_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'city_name' in daily_weather_df.columns:
                    num_cities = daily_weather_df['city_name'].nunique()
                    st.metric("Количество городов", num_cities)
                else:
                    st.metric("Всего записей", len(daily_weather_df))
            
            with col2:
                if 'date' in daily_weather_df.columns:
                    date_range_str = f"{daily_weather_df['date'].min()} - {daily_weather_df['date'].max()}"
                    st.metric("Диапазон дат", date_range_str[:20] + "..." if len(date_range_str) > 20 else date_range_str)
                else:
                    st.metric("Числовых признаков", len(daily_weather_df.select_dtypes(include=[np.number]).columns))
            
            with col3:
                numeric_cols = daily_weather_df.select_dtypes(include=[np.number]).columns
                st.metric("Числовых признаков", len(numeric_cols))
            
            with col4:
                missing_total = daily_weather_df.isnull().sum().sum()
                st.metric("Пропущенных значений", missing_total)
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📋 Данные", "📊 Графики", "🌍 Карта"])
        
        with tab1:
            st.header("Просмотр данных")
            
            dataset_choice = st.selectbox(
                "Выберите датасет:",
                ["Ежедневные данные", "Города", "Страны"]
            )
            
            if dataset_choice == "Ежедневные данные" and not daily_weather_df.empty:
                df_display = daily_weather_df
            elif dataset_choice == "Города" and not cities_weather_df.empty:
                df_display = cities_weather_df
            elif dataset_choice == "Страны" and not countries_weather_df.empty:
                df_display = countries_weather_df
            else:
                st.warning("Данный датасет не загружен")
                df_display = pd.DataFrame()
            
            if not df_display.empty:
                # Поиск
                search_col1, search_col2 = st.columns([2, 1])
                with search_col1:
                    search_term = st.text_input("Поиск по таблице:", "")
                
                with search_col2:
                    rows_per_page = st.selectbox("Строк на странице:", [10, 25, 50, 100], index=0)
                
                # Применение поиска
                if search_term:
                    mask = df_display.apply(
                        lambda row: row.astype(str).str.contains(search_term, case=False, na=False).any(),
                        axis=1
                    )
                    df_filtered = df_display[mask]
                else:
                    df_filtered = df_display
                
                # Пагинация
                total_pages = max(1, len(df_filtered) // rows_per_page + (1 if len(df_filtered) % rows_per_page > 0 else 0))
                page_number = st.number_input("Страница:", min_value=1, max_value=total_pages, value=1)
                
                start_idx = (page_number - 1) * rows_per_page
                end_idx = min(start_idx + rows_per_page, len(df_filtered))
                
                # Отображение
                st.dataframe(
                    df_filtered.iloc[start_idx:end_idx],
                    use_container_width=True,
                    height=400
                )
                
                st.caption(f"Показано {start_idx+1}-{end_idx} из {len(df_filtered)} записей")
                
                # Статистика
                if st.checkbox("Показать статистику"):
                    numeric_cols = df_display.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.subheader("Статистика числовых признаков")
                        st.dataframe(df_display[numeric_cols].describe())
        
        with tab2:
            st.header("Визуализация данных")
            
            if not daily_weather_df.empty:
                # Выбор типа графика
                chart_type = st.selectbox(
                    "Тип графика:",
                    ["Гистограмма", "Box Plot", "Scatter Plot", "Линейный график", "Heatmap корреляций"]
                )
                
                numeric_cols = daily_weather_df.select_dtypes(include=[np.number]).columns.tolist()
                
                if chart_type == "Гистограмма":
                    col_selected = st.selectbox("Выберите колонку:", numeric_cols)
                    fig = px.histogram(daily_weather_df, x=col_selected, nbins=50,
                                      title=f"Распределение {col_selected}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Box Plot":
                    col_selected = st.selectbox("Выберите колонку:", numeric_cols)
                    fig = px.box(daily_weather_df, y=col_selected, title=f"Box Plot: {col_selected}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Scatter Plot":
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("X ось:", numeric_cols)
                    with col2:
                        y_col = st.selectbox("Y ось:", numeric_cols)
                    
                    color_by = None
                    if 'city_name' in daily_weather_df.columns:
                        color_by = st.selectbox("Цвет по:", ['Нет'] + ['city_name'])
                        color_by = None if color_by == 'Нет' else color_by
                    
                    fig = px.scatter(daily_weather_df, x=x_col, y=y_col, color=color_by,
                                    title=f"{y_col} vs {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Линейный график":
                    if 'date' in daily_weather_df.columns:
                        # Выбор города для фильтрации
                        if 'city_name' in daily_weather_df.columns:
                            city_filter = st.selectbox("Выберите город:", 
                                                      ['Все'] + daily_weather_df['city_name'].unique().tolist())
                            if city_filter != 'Все':
                                df_chart = daily_weather_df[daily_weather_df['city_name'] == city_filter]
                            else:
                                df_chart = daily_weather_df
                        else:
                            df_chart = daily_weather_df
                        
                        # Выбор переменной
                        y_col = st.selectbox("Выберите переменную:", numeric_cols)
                        
                        # Агрегация по дате
                        df_agg = df_chart.groupby('date')[y_col].mean().reset_index()
                        
                        fig = px.line(df_agg, x='date', y=y_col,
                                     title=f"{y_col} по времени")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Для линейного графика нужна колонка с датами")
                
                elif chart_type == "Heatmap корреляций":
                    if len(numeric_cols) > 1:
                        corr_matrix = daily_weather_df[numeric_cols].corr()
                        
                        fig = px.imshow(corr_matrix,
                                       text_auto=True,
                                       aspect="auto",
                                       title="Матрица корреляций",
                                       color_continuous_scale='RdBu_r')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Топ корреляции
                        st.subheader("Наиболее сильные корреляции")
                        corr_pairs = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                corr = corr_matrix.iloc[i, j]
                                if abs(corr) > 0.5:
                                    corr_pairs.append({
                                        'Признак 1': corr_matrix.columns[i],
                                        'Признак 2': corr_matrix.columns[j],
                                        'Корреляция': corr
                                    })
                        
                        if corr_pairs:
                            corr_df = pd.DataFrame(corr_pairs)
                            st.dataframe(corr_df.sort_values('Корреляция', key=abs, ascending=False))
                        else:
                            st.info("Сильных корреляций (|r| > 0.5) не найдено")
        
        with tab3:
            st.header("Географическая визуализация")
            
            if not cities_weather_df.empty and 'latitude' in cities_weather_df.columns and 'longitude' in cities_weather_df.columns:
                # Выбор атрибута для цветового кодирования
                numeric_cols_cities = cities_weather_df.select_dtypes(include=[np.number]).columns.tolist()
                
                color_attribute = st.selectbox(
                    "Цвет по атрибуту:",
                    ['Нет'] + numeric_cols_cities
                )
                
                size_attribute = st.selectbox(
                    "Размер по атрибуту:",
                    ['Нет'] + numeric_cols_cities
                )
                
                # Подготовка данных для карты
                map_data = cities_weather_df.copy()
                
                # Создание карты
                if color_attribute != 'Нет':
                    fig = px.scatter_geo(map_data,
                                        lat='latitude',
                                        lon='longitude',
                                        color=color_attribute,
                                        size=size_attribute if size_attribute != 'Нет' else None,
                                        hover_name='city_name' if 'city_name' in map_data.columns else None,
                                        title='Распределение городов',
                                        projection='natural earth')
                else:
                    fig = px.scatter_geo(map_data,
                                        lat='latitude',
                                        lon='longitude',
                                        size=size_attribute if size_attribute != 'Нет' else None,
                                        hover_name='city_name' if 'city_name' in map_data.columns else None,
                                        title='Распределение городов',
                                        projection='natural earth')
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Для географической визуализации нужны данные с координатами (latitude, longitude)")

# ==========================================================
# PAGE 2 — ANALYSIS
# ==========================================================
elif page == "🔍 Анализ":
    
    st.title("🔍 Анализ погодных данных")
    
    # Выбор метода анализа
    analysis_method = st.selectbox(
        "Метод анализа:",
        ["Кластеризация", "Регрессия", "Временные ряды", "Анализ главных компонент (PCA)"]
    )
    
    # Используем только daily_weather_df для анализа
    if daily_weather_df.empty:
        st.error("Для анализа нужны данные. Загрузите daily_weather.csv")
    else:
        # Предобработка данных для анализа
        df_analysis = daily_weather_df.copy()
        
        # Выбор числовых колонок
        numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.error("Нет числовых колонок для анализа")
        else:
            # Стандартизация
            scaler = StandardScaler()
            df_scaled = df_analysis.copy()
            df_scaled[numeric_cols] = scaler.fit_transform(df_analysis[numeric_cols].fillna(0))
            
            # ========== КЛАСТЕРИЗАЦИЯ ==========
            if analysis_method == "Кластеризация":
                st.header("Кластеризация данных")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Выбор признаков
                    features = st.multiselect(
                        "Выберите признаки для кластеризации:",
                        numeric_cols,
                        default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
                    )
                
                with col2:
                    # Выбор алгоритма
                    algorithm = st.selectbox("Алгоритм:", ["K-Means", "DBSCAN"])
                    
                    if algorithm == "K-Means":
                        n_clusters = st.slider("Количество кластеров:", 2, 10, 3)
                    else:
                        eps_value = st.slider("EPS:", 0.1, 2.0, 0.5, 0.1)
                        min_samples_value = st.slider("Минимум образцов:", 1, 20, 5)
                
                if len(features) >= 2:
                    # Подготовка данных
                    X = df_scaled[features].fillna(0)
                    
                    if algorithm == "K-Means":
                        # K-Means кластеризация
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(X)
                        
                        # Метрики
                        inertia = kmeans.inertia_
                        try:
                            silhouette = silhouette_score(X, clusters)
                        except:
                            silhouette = None
                        
                        # Отображение метрик
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Количество кластеров", n_clusters)
                        with col2:
                            st.metric("Inertia", f"{inertia:.2f}")
                        with col3:
                            if silhouette:
                                st.metric("Silhouette Score", f"{silhouette:.3f}")
                            else:
                                st.metric("Silhouette Score", "N/A")
                        
                        centers = kmeans.cluster_centers_
                        
                    else:
                        # DBSCAN кластеризация
                        dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
                        clusters = dbscan.fit_predict(X)
                        
                        # Статистика кластеров
                        unique_clusters = set(clusters)
                        n_clusters = len(unique_clusters) - (1 if -1 in clusters else 0)
                        noise_points = sum(clusters == -1)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Обнаружено кластеров", n_clusters)
                        with col2:
                            st.metric("Точек шума", noise_points)
                    
                    # Визуализация
                    df_viz = df_analysis.copy()
                    df_viz['Cluster'] = clusters
                    
                    # 2D scatter plot
                    if len(features) >= 2:
                        fig = px.scatter(
                            df_viz,
                            x=features[0],
                            y=features[1],
                            color='Cluster',
                            title=f"Кластеризация: {features[0]} vs {features[1]}",
                            hover_data=['city_name'] if 'city_name' in df_viz.columns else None
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Анализ кластеров
                    st.subheader("📊 Характеристики кластеров")
                    
                    if 'Cluster' in df_viz.columns:
                        cluster_stats = df_viz.groupby('Cluster')[numeric_cols].mean()
                        st.dataframe(cluster_stats.style.background_gradient(cmap='coolwarm'))
                        
                        # Распределение по кластерам
                        cluster_counts = df_viz['Cluster'].value_counts().sort_index()
                        fig_counts = px.bar(
                            x=cluster_counts.index.astype(str),
                            y=cluster_counts.values,
                            title="Распределение точек по кластерам",
                            labels={'x': 'Кластер', 'y': 'Количество точек'}
                        )
                        st.plotly_chart(fig_counts, use_container_width=True)
            
            # ========== РЕГРЕССИЯ ==========
            elif analysis_method == "Регрессия":
                st.header("Регрессионный анализ")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Выбор целевой переменной
                    target = st.selectbox("Целевая переменная (Y):", numeric_cols)
                
                with col2:
                    # Выбор признаков
                    available_features = [col for col in numeric_cols if col != target]
                    features = st.multiselect(
                        "Признаки (X):",
                        available_features,
                        default=available_features[:3] if len(available_features) >= 3 else available_features
                    )
                
                if target and features:
                    # Подготовка данных
                    X = df_scaled[features].fillna(0)
                    y = df_scaled[target]
                    
                    # Разделение на train/test
                    test_size = st.slider("Размер тестовой выборки (%):", 10, 50, 20)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=42
                    )
                    
                    # Выбор модели
                    model_type = st.selectbox(
                        "Тип модели:",
                        ["Линейная регрессия", "Ridge", "Lasso", "Случайный лес"]
                    )
                    
                    if model_type == "Линейная регрессия":
                        model = LinearRegression()
                    elif model_type == "Ridge":
                        alpha = st.slider("Alpha (регуляризация):", 0.01, 10.0, 1.0)
                        model = Ridge(alpha=alpha)
                    elif model_type == "Lasso":
                        alpha = st.slider("Alpha (регуляризация):", 0.01, 10.0, 1.0)
                        model = Lasso(alpha=alpha)
                    else:
                        n_estimators = st.slider("Количество деревьев:", 10, 200, 100)
                        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                    
                    # Обучение модели
                    model.fit(X_train, y_train)
                    
                    # Прогнозы
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    # Метрики
                    r2_train = r2_score(y_train, y_pred_train)
                    r2_test = r2_score(y_test, y_pred_test)
                    mae_train = mean_absolute_error(y_train, y_pred_train)
                    mae_test = mean_absolute_error(y_test, y_pred_test)
                    
                    # Отображение метрик
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("R² (обучение)", f"{r2_train:.3f}")
                    with col2:
                        st.metric("R² (тест)", f"{r2_test:.3f}")
                    with col3:
                        st.metric("MAE (обучение)", f"{mae_train:.3f}")
                    with col4:
                        st.metric("MAE (тест)", f"{mae_test:.3f}")
                    
                    # Визуализация результатов
                    tab1, tab2 = st.tabs(["📈 Прогнозы", "📊 Важность признаков"])
                    
                    with tab1:
                        # График фактических vs предсказанных значений
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=y_test,
                            y=y_pred_test,
                            mode='markers',
                            name='Тестовая выборка',
                            marker=dict(color='blue', opacity=0.6)
                        ))
                        
                        # Линия идеального прогноза
                        min_val = min(y_test.min(), y_pred_test.min())
                        max_val = max(y_test.max(), y_pred_test.max())
                        fig.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Идеальный прогноз',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title="Фактические vs Предсказанные значения",
                            xaxis_title="Фактические значения",
                            yaxis_title="Предсказанные значения"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        # Важность признаков
                        if hasattr(model, 'feature_importances_'):
                            importance_df = pd.DataFrame({
                                'Признак': features,
                                'Важность': model.feature_importances_
                            }).sort_values('Важность', ascending=False)
                            
                            fig = px.bar(
                                importance_df,
                                x='Признак',
                                y='Важность',
                                title="Важность признаков в модели"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        elif hasattr(model, 'coef_'):
                            coef_df = pd.DataFrame({
                                'Признак': features,
                                'Коэффициент': model.coef_
                            }).sort_values('Коэффициент', ascending=False)
                            
                            fig = px.bar(
                                coef_df,
                                x='Признак',
                                y='Коэффициент',
                                title="Коэффициенты линейной модели"
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
            # ========== ВРЕМЕННЫЕ РЯДЫ ==========
            elif analysis_method == "Временные ряды":
                st.header("Анализ временных рядов")
                
                if 'date' not in df_analysis.columns:
                    st.warning("Для анализа временных рядов нужна колонка с датами.")
                else:
                    # Выбор города
                    if 'city_name' in df_analysis.columns:
                        city = st.selectbox("Выберите город:", 
                                          ['Все города'] + df_analysis['city_name'].unique().tolist())
                        if city != 'Все города':
                            df_city = df_analysis[df_analysis['city_name'] == city]
                        else:
                            df_city = df_analysis
                    else:
                        df_city = df_analysis
                        city = "Все данные"
                    
                    # Выбор переменной
                    variable = st.selectbox("Выберите переменную:", numeric_cols)
                    
                    # Агрегация по дате
                    df_ts = df_city.groupby('date')[variable].mean().reset_index()
                    df_ts = df_ts.sort_values('date')
                    
                    # Визуализация временного ряда
                    fig = px.line(
                        df_ts,
                        x='date',
                        y=variable,
                        title=f"{variable} по времени для {city}",
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Упрощенный анализ тренда
                    st.subheader("📊 Анализ тренда")
                    
                    window_size = st.slider("Окно для скользящего среднего:", 7, 90, 30)
                    
                    df_ts['moving_avg'] = df_ts[variable].rolling(
                        window=min(window_size, len(df_ts)), 
                        center=True, 
                        min_periods=1
                    ).mean()
                    
                    fig_trend = go.Figure()
                    
                    fig_trend.add_trace(go.Scatter(
                        x=df_ts['date'],
                        y=df_ts[variable],
                        name='Исходные данные',
                        line=dict(color='lightblue', width=1)
                    ))
                    
                    fig_trend.add_trace(go.Scatter(
                        x=df_ts['date'],
                        y=df_ts['moving_avg'],
                        name=f'Скользящее среднее ({window_size} дней)',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig_trend.update_layout(
                        title=f"Тренд {variable} для {city}",
                        xaxis_title="Дата",
                        yaxis_title=variable
                    )
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Автокорреляция
                    st.subheader("📈 Автокорреляция")
                    
                    autocorr_values = calculate_autocorrelation(df_ts[variable], max_lags=50)
                    
                    if autocorr_values:
                        fig_acf = go.Figure()
                        
                        fig_acf.add_trace(go.Bar(
                            x=list(range(1, len(autocorr_values) + 1)),
                            y=autocorr_values,
                            name='Автокорреляция'
                        ))
                        
                        # Доверительные интервалы
                        conf_int = 1.96 / np.sqrt(len(df_ts))
                        fig_acf.add_hline(y=conf_int, line_dash="dash", line_color="red",
                                         annotation_text="Доверительный интервал 95%")
                        fig_acf.add_hline(y=-conf_int, line_dash="dash", line_color="red")
                        
                        fig_acf.update_layout(
                            title="Функция автокорреляции (ACF)",
                            xaxis_title="Лаг (дни)",
                            yaxis_title="Автокорреляция"
                        )
                        
                        st.plotly_chart(fig_acf, use_container_width=True)
            
            # ========== PCA ==========
            elif analysis_method == "Анализ главных компонент (PCA)":
                st.header("Анализ главных компонент (PCA)")
                
                # Выбор признаков
                pca_features = st.multiselect(
                    "Выберите признаки для PCA:",
                    numeric_cols,
                    default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
                )
                
                if len(pca_features) >= 2:
                    n_components = st.slider(
                        "Количество компонент:",
                        2, min(10, len(pca_features)), 3
                    )
                    
                    X_pca = df_scaled[pca_features].fillna(0)
                    
                    # Применение PCA
                    pca = PCA(n_components=n_components)
                    X_pca_transformed = pca.fit_transform(X_pca)
                    
                    # Объясненная дисперсия
                    explained_variance = pca.explained_variance_ratio_
                    cumulative_variance = explained_variance.cumsum()
                    
                    # График объясненной дисперсии
                    fig_var = go.Figure()
                    
                    fig_var.add_trace(go.Bar(
                        x=[f"PC{i+1}" for i in range(n_components)],
                        y=explained_variance,
                        name='Доля дисперсии'
                    ))
                    
                    fig_var.add_trace(go.Scatter(
                        x=[f"PC{i+1}" for i in range(n_components)],
                        y=cumulative_variance,
                        name='Накопленная дисперсия',
                        yaxis='y2'
                    ))
                    
                    fig_var.update_layout(
                        title="Объясненная дисперсия по компонентам",
                        yaxis=dict(title='Доля дисперсии'),
                        yaxis2=dict(
                            title='Накопленная дисперсия',
                            overlaying='y',
                            side='right'
                        )
                    )
                    
                    st.plotly_chart(fig_var, use_container_width=True)
                    
                    # 2D визуализация PCA
                    if n_components >= 2:
                        df_pca_viz = df_analysis.copy()
                        df_pca_viz['PC1'] = X_pca_transformed[:, 0]
                        df_pca_viz['PC2'] = X_pca_transformed[:, 1]
                        
                        # Выбор атрибута для цвета
                        color_options = ['Нет'] + pca_features
                        if 'city_name' in df_pca_viz.columns:
                            color_options.append('city_name')
                        
                        color_by = st.selectbox("Цвет по:", color_options)
                        
                        fig_pca = px.scatter(
                            df_pca_viz,
                            x='PC1',
                            y='PC2',
                            color=None if color_by == 'Нет' else color_by,
                            title="PCA - Первые две главные компоненты",
                            hover_data=pca_features
                        )
                        
                        st.plotly_chart(fig_pca, use_container_width=True)
                    
                    # Нагрузки компонент
                    st.subheader("📊 Нагрузки компонент")
                    
                    loadings = pd.DataFrame(
                        pca.components_.T,
                        columns=[f'PC{i+1}' for i in range(n_components)],
                        index=pca_features
                    )
                    
                    st.dataframe(loadings.style.background_gradient(cmap='coolwarm', axis=0))

# ==========================================================
# PAGE 3 — FORECASTING
# ==========================================================
elif page == "📈 Прогнозирование":
    
    st.title("📈 Прогнозирование погодных данных")
    
    if daily_weather_df.empty or 'date' not in daily_weather_df.columns:
        st.warning("Для прогнозирования нужны данные с временными метками.")
    else:
        st.info("""
        Этот инструмент позволяет строить простые прогнозы на основе исторических данных.
        Используется метод скользящего среднего с линейной экстраполяцией.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Выбор города
            if 'city_name' in daily_weather_df.columns:
                forecast_city = st.selectbox(
                    "Город:",
                    daily_weather_df['city_name'].unique()
                )
                df_city = daily_weather_df[daily_weather_df['city_name'] == forecast_city]
            else:
                forecast_city = "Все данные"
                df_city = daily_weather_df
        
        with col2:
            # Выбор переменной
            numeric_cols = df_city.select_dtypes(include=[np.number]).columns.tolist()
            forecast_variable = st.selectbox(
                "Переменная для прогноза:",
                numeric_cols
            )
        
        with col3:
            # Горизонт прогноза
            forecast_days = st.slider("Дней прогноза:", 1, 90, 30)
        
        # Подготовка временного ряда
        df_city['date'] = pd.to_datetime(df_city['date'])
        df_ts = df_city.groupby('date')[forecast_variable].mean().reset_index()
        df_ts = df_ts.sort_values('date')
        
        if len(df_ts) > 0:
            # Простой прогноз на основе тренда
            st.subheader("📊 Простой прогноз")
            
            # Размер окна для скользящего среднего
            window = st.slider("Размер окна для анализа тренда:", 7, 365, 30)
            
            # Вычисляем скользящее среднее
            df_ts['moving_avg'] = df_ts[forecast_variable].rolling(window=window, center=True).mean()
            
            # Линейная регрессия для экстраполяции
            from sklearn.linear_model import LinearRegression
            
            # Используем последние N точек для прогноза
            last_n_points = min(100, len(df_ts))
            recent_data = df_ts.tail(last_n_points).copy()
            recent_data['days'] = np.arange(len(recent_data))
            
            # Обучение модели
            X = recent_data[['days']]
            y = recent_data[forecast_variable]
            model = LinearRegression()
            model.fit(X, y)
            
            # Прогноз
            future_days = np.arange(len(recent_data), len(recent_data) + forecast_days).reshape(-1, 1)
            forecast_values = model.predict(future_days)
            
            # Даты для прогноза
            last_date = recent_data['date'].iloc[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Создаем DataFrame с прогнозом
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                forecast_variable: forecast_values,
                'type': 'Прогноз'
            })
            
            # Подготовка исторических данных
            history_df = df_ts[['date', forecast_variable]].copy()
            history_df['type'] = 'История'
            
            # Объединяем
            combined_df = pd.concat([history_df, forecast_df])
            
            # Визуализация
            fig = px.line(
                combined_df,
                x='date',
                y=forecast_variable,
                color='type',
                title=f"Прогноз {forecast_variable} для {forecast_city}",
                markers=True
            )
            
            # Добавляем доверительный интервал
            historical_std = df_ts[forecast_variable].std()
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_dates, forecast_dates[::-1]]),
                y=pd.concat([
                    forecast_values + historical_std,
                    (forecast_values - historical_std)[::-1]
                ]),
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.2)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name='Доверительный интервал (1σ)',
                showlegend=True
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Статистика прогноза
            st.subheader("📈 Статистика прогноза")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Среднее прогнозное значение", f"{forecast_values.mean():.2f}")
            with col2:
                st.metric("Минимум прогноза", f"{forecast_values.min():.2f}")
            with col3:
                st.metric("Максимум прогноза", f"{forecast_values.max():.2f}")
            with col4:
                trend = "📈 Восходящий" if model.coef_[0] > 0 else "📉 Нисходящий" if model.coef_[0] < 0 else "➡️ Стабильный"
                st.metric("Тренд", trend)
            
            # Скачивание прогноза
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Скачать прогноз",
                data=csv,
                file_name=f"forecast_{forecast_city}_{forecast_variable}.csv",
                mime="text/csv"
            )

# ==========================================================
# PAGE 4 — ABOUT
# ==========================================================
else:
    
    st.title("ℹ️ О проекте")
    
    st.markdown("""
    # Weather Analytics Dashboard
    
    ## 📊 Описание
    Интерактивный дашборд для анализа глобальных ежедневных погодных данных.
    
    ## 🔧 Функционал
    1. **Визуализация данных** - Исследование распределений, корреляций и географических данных
    2. **Анализ** - Кластеризация, регрессия, анализ временных рядов, PCA
    3. **Прогнозирование** - Простые прогнозы погодных параметров
    
    ## 📁 Используемые данные
    Проект работает с тремя основными датасетами:
    
    ### 1. Daily Weather Data
    - Ежедневные погодные измерения
    - Температура, осадки, влажность, скорость ветра
    - Временные ряды для анализа трендов
    
    ### 2. Cities Data
    - Географическая информация о городах
    - Координаты (широта/долгота)
    - Демографические данные
    
    ### 3. Countries Data
    - Климатические характеристики стран
    - Средние показатели по странам
    
    ## 🛠️ Технологии
    - **Streamlit** - Фреймворк для создания веб-приложений
    - **Plotly** - Интерактивная визуализация
    - **Scikit-learn** - Машинное обучение и анализ данных
    - **Pandas & NumPy** - Обработка и анализ данных
    - **Seaborn & Matplotlib** - Статичная визуализация
    
    ## 📊 Методы анализа
    ### 1. Кластеризация
    - K-Means для группировки похожих наблюдений
    - DBSCAN для обнаружения кластеров произвольной формы
    - Визуализация кластеров в 2D пространстве
    
    ### 2. Регрессионный анализ
    - Линейная регрессия
    - Ridge и Lasso регрессия
    - Случайный лес для нелинейных зависимостей
    - Оценка важности признаков
    
    ### 3. Анализ временных рядов
    - Визуализация трендов
    - Скользящее среднее
    - Автокорреляционный анализ
    - Простое прогнозирование
    
    ### 4. PCA (Анализ главных компонент)
    - Снижение размерности
    - Выявление скрытых паттернов
    - Визуализация в уменьшенном пространстве
    
    ## 🚀 Запуск проекта
    
    ### Локальный запуск
    ```bash
    # 1. Установите зависимости
    pip install -r requirements.txt
    
    # 2. Запустите приложение
    streamlit run app.py
    ```
    
    ### Требования к данным
    Поместите CSV файлы в корневую директорию:
    - `daily_weather_smallest.csv` или `daily_weather.csv`
    - `cities.csv` или `cities_weather.csv`
    - `countries.csv` или `countries_weather.csv`
    
    ## 📱 Особенности интерфейса
    - **Адаптивный дизайн** - Работает на компьютерах и мобильных устройствах
    - **Интерактивность** - Фильтры, сортировка, поиск в реальном времени
    - **Профессиональная визуализация** - Интерактивные графики с подсказками
    - **Экспорт данных** - Возможность скачивания результатов анализа
    
    ## 🎯 Цели проекта
    1. Предоставить удобный инструмент для анализа погодных данных
    2. Демонстрация различных методов анализа данных
    3. Создание интерактивной платформы для исследования климатических паттернов
    4. Образовательная цель - показать применение ML в климатологии
    
    ## 📞 Контакты
    Для вопросов и предложений: [ваш-email@example.com](mailto:ваш-email@example.com)
    
    ## 📄 Лицензия
    Проект распространяется под лицензией MIT.
    
    ---
    
    *Последнее обновление: {}
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M')))

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption("🌤️ Weather Analytics Dashboard")
with footer_col2:
    st.caption(f"Версия 2.0 • Данные обновлены: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
with footer_col3:
    st.caption("© 2024 Все права защищены")

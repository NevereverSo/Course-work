import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML библиотеки
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Weather Analytics Dashboard", 
    layout="wide"
)

# ----------------------------------------------------------
# LOAD DATA С ОПТИМИЗАЦИЕЙ
# ----------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Загрузка данных...")
def load_and_preprocess_data():
    """Загрузка и предобработка данных с кэшированием"""
    try:
        daily_df = pd.read_csv("daily_weather_smallest.csv", low_memory=False)
        if not daily_df.empty and 'date' in daily_df.columns:
            daily_df['date'] = pd.to_datetime(daily_df['date'], errors='coerce')
    except Exception as e:
        st.sidebar.error(f"Ошибка загрузки daily: {str(e)}")
        daily_df = pd.DataFrame()
    
    try:
        cities_df = pd.read_csv("cities.csv", low_memory=False)
    except:
        cities_df = pd.DataFrame()
    
    try:
        countries_df = pd.read_csv("countries.csv", low_memory=False)
    except:
        countries_df = pd.DataFrame()
    
    return countries_df, cities_df, daily_df

# Загрузка данных
countries_df, cities_df, daily_df = load_and_preprocess_data()

# ----------------------------------------------------------
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ С КЭШЕМ
# ----------------------------------------------------------
@st.cache_data
def get_numeric_columns(df):
    """Получение числовых колонок, исключая station_id и другие идентификаторы"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Убираем station_id и другие идентификаторы из числовых признаков
    id_columns = ['station_id', 'id', 'station', 'station_code', 'station_number']
    
    # Фильтруем только те колонки, которые реально являются числовыми признаками
    filtered_cols = []
    for col in numeric_cols:
        # Проверяем, не является ли колонка идентификатором
        is_id_column = False
        for id_pattern in id_columns:
            if id_pattern in col.lower():
                is_id_column = True
                break
        
        # Также проверяем, не слишком ли много уникальных значений (возможно, это ID)
        if not is_id_column and col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.95:  # Если менее 95% уникальных значений - вероятно, не ID
                filtered_cols.append(col)
    
    return filtered_cols

@st.cache_data
def prepare_scaled_data(_df, numeric_cols):
    """Подготовка масштабированных данных с кэшированием"""
    scaler = StandardScaler()
    df_scaled = _df.copy()
    
    # Заполняем пропуски перед масштабированием
    df_filled = _df[numeric_cols].fillna(_df[numeric_cols].mean())
    df_scaled[numeric_cols] = scaler.fit_transform(df_filled)
    
    return df_scaled

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
st.sidebar.title("Weather Analytics")

# Навигация
page = st.sidebar.radio(
    "Навигация",
    ["Визуализация данных", "Анализ данных"]
)

# Статус данных
if not daily_df.empty:
    st.sidebar.success(f"Данные загружены: {len(daily_df):,} записей")
    
    # Показываем информацию о колонках
    st.sidebar.markdown("---")
    st.sidebar.subheader("Информация о данных")
    
    numeric_cols = get_numeric_columns(daily_df)
    st.sidebar.info(f"Числовых признаков: {len(numeric_cols)}")
    
    # Показываем список числовых признаков (первые 10)
    if numeric_cols:
        st.sidebar.write("**Числовые признаки:**")
        for col in numeric_cols[:10]:
            st.sidebar.write(f"• {col}")
        if len(numeric_cols) > 10:
            st.sidebar.write(f"• ... и еще {len(numeric_cols) - 10}")
    
    # Показываем нечисловые колонки
    non_numeric_cols = [col for col in daily_df.columns if col not in numeric_cols]
    if non_numeric_cols:
        st.sidebar.write("**Нечисловые колонки:**")
        for col in non_numeric_cols[:5]:
            st.sidebar.write(f"• {col}")
        if len(non_numeric_cols) > 5:
            st.sidebar.write(f"• ... и еще {len(non_numeric_cols) - 5}")
else:
    st.sidebar.error("Данные не загружены")

if page == "Визуализация данных":
    st.title("Визуализация погодных данных")
else:
    st.title("Анализ погодных данных")

# ==========================================================
# PAGE 1 — ВИЗУАЛИЗАЦИЯ ДАННЫХ
# ==========================================================
if page == "Визуализация данных":
    
    if daily_df.empty:
        st.error("Данные не загружены. Убедитесь, что файл daily_weather_smallest.csv находится в корневой директории.")
    else:
        # Получаем числовые колонки (без station_id)
        numeric_cols = get_numeric_columns(daily_df)
        
        # Быстрые KPI метрики
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'city_name' in daily_df.columns:
                st.metric("Городов", daily_df['city_name'].nunique())
            else:
                st.metric("Записей", len(daily_df))
        
        with col2:
            if 'date' in daily_df.columns:
                date_range = f"{daily_df['date'].min().date()} - {daily_df['date'].max().date()}"
                st.metric("Период", date_range)
        
        with col3:
            st.metric("Числовых признаков", len(numeric_cols))
        
        with col4:
            st.metric("Пропусков", int(daily_df[numeric_cols].isnull().sum().sum()))
        
        # Вкладки
        tab1, tab2, tab3, tab4 = st.tabs(["Данные", "Распределения", "Корреляции", "География"])
        
        with tab1:
            st.header("Просмотр данных")
            
            dataset_choice = st.selectbox(
                "Выберите датасет:",
                ["Ежедневные данные", "Города", "Страны"]
            )
            
            if dataset_choice == "Ежедневные данные":
                df_display = daily_df
                # Быстрый фильтр по городу
                if 'city_name' in daily_df.columns:
                    selected_city = st.selectbox(
                        "Фильтр по городу:", 
                        ['Все'] + daily_df['city_name'].unique().tolist()[:10]
                    )
                    if selected_city != 'Все':
                        df_display = df_display[df_display['city_name'] == selected_city]
            
            elif dataset_choice == "Города" and not cities_df.empty:
                df_display = cities_df
            elif dataset_choice == "Страны" and not countries_df.empty:
                df_display = countries_df
            else:
                st.warning("Датасет не загружен")
                df_display = pd.DataFrame()
            
            if not df_display.empty:
                # Показываем только первые 500 строк для скорости
                preview_rows = st.slider("Показать строк:", 100, 1000, 500, step=100)
                st.dataframe(df_display.head(preview_rows), use_container_width=True)
                
                # Быстрая статистика (только для числовых признаков)
                if st.checkbox("Показать описательную статистику"):
                    if dataset_choice == "Ежедневные данные":
                        display_cols = get_numeric_columns(df_display)
                    else:
                        display_cols = df_display.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if display_cols:
                        st.dataframe(df_display[display_cols].describe(), use_container_width=True)
        
        with tab2:
            st.header("Анализ распределений")
            
            if numeric_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_col = st.selectbox("Выберите признак:", numeric_cols)
                    plot_type = st.radio(
                        "Тип графика:",
                        ["Гистограмма", "Box Plot", "Violin Plot"]
                    )
                
                # Убрана опция "Цвет по городу" - она не имеет смысла при анализе распределений
                
                # Быстрое создание графиков
                if plot_type == "Гистограмма":
                    fig = px.histogram(
                        daily_df, 
                        x=selected_col, 
                        nbins=50,
                        title=f"Распределение {selected_col}"
                    )
                elif plot_type == "Box Plot":
                    fig = px.box(
                        daily_df, 
                        y=selected_col,
                        title=f"Box Plot: {selected_col}"
                    )
                else:
                    fig = px.violin(
                        daily_df, 
                        y=selected_col,
                        title=f"Violin Plot: {selected_col}"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Нет числовых признаков для анализа распределений")
        
        with tab3:
            st.header("Корреляционный анализ")
            
            if len(numeric_cols) > 1:
                # Ограничиваем для скорости
                max_features = min(10, len(numeric_cols))
                selected_features = st.multiselect(
                    "Выберите признаки для анализа:",
                    numeric_cols,
                    default=numeric_cols[:max_features]
                )
                
                if len(selected_features) > 1:
                    # Вычисляем корреляции
                    corr_matrix = daily_df[selected_features].corr()
                    
                    # Heatmap
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=".2f",
                        aspect="auto",
                        title="Корреляционная матрица",
                        color_continuous_scale="RdBu_r"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Scatter matrix для выбранных признаков
                    if len(selected_features) <= 5:
                        fig = px.scatter_matrix(
                            daily_df[selected_features],
                            title="Матрица рассеяния"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Нужно минимум 2 числовых признака для корреляционного анализа")
        
        with tab4:
            st.header("Географическая визуализация")
            
            if not cities_df.empty and 'latitude' in cities_df.columns and 'longitude' in cities_df.columns:
                # Получаем числовые колонки для cities
                numeric_cols_cities = get_numeric_columns(cities_df)
                
                if numeric_cols_cities:
                    color_by = st.selectbox(
                        "Цвет по признаку:",
                        ['Нет'] + numeric_cols_cities
                    )
                    
                    size_by = st.selectbox(
                        "Размер по признаку:",
                        ['Нет'] + numeric_cols_cities
                    )
                    
                    fig = px.scatter_geo(
                        cities_df,
                        lat='latitude',
                        lon='longitude',
                        color=color_by if color_by != 'Нет' else None,
                        size=size_by if size_by != 'Нет' else None,
                        hover_name='city_name' if 'city_name' in cities_df.columns else None,
                        title="Распределение городов"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.scatter_geo(
                        cities_df,
                        lat='latitude',
                        lon='longitude',
                        hover_name='city_name' if 'city_name' in cities_df.columns else None,
                        title="Распределение городов"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Для географической визуализации нужны данные городов с координатами")

# ==========================================================
# PAGE 2 — АНАЛИЗ ДАННЫХ
# ==========================================================
else:
    
    if daily_df.empty:
        st.error("Для анализа нужны ежедневные данные")
    else:
        # Получаем числовые колонки (без station_id)
        numeric_cols = get_numeric_columns(daily_df)
        
        if not numeric_cols:
            st.error("Нет числовых признаков для анализа. Проверьте данные.")
        else:
            # Показываем доступные признаки
            st.info(f"**Доступно для анализа:** {len(numeric_cols)} числовых признаков")
            
            # Выбор метода анализа
            analysis_method = st.selectbox(
                "Метод анализа:",
                ["Кластеризация", "Регрессия", "Временные ряды", "PCA"]
            )
            
            # Подготовка масштабированных данных (с кэшированием)
            df_scaled = prepare_scaled_data(daily_df, numeric_cols)
            
                       # ========== КЛАСТЕРИЗАЦИЯ ==========
            if analysis_method == "Кластеризация":
                st.header("Кластеризация данных")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    features = st.multiselect(
                        "Выберите признаки для кластеризации:",
                        numeric_cols,
                        default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
                    )
                
                with col2:
                    algorithm = st.selectbox("Алгоритм:", ["K-Means", "DBSCAN"])
                    
                    if algorithm == "K-Means":
                        n_clusters = st.slider("Кластеров:", 2, 10, 3)
                    else:
                        eps = st.slider("EPS:", 0.1, 2.0, 0.5, 0.1)
                        min_samples = st.slider("Min samples:", 2, 20, 5)
                
                if len(features) >= 2:
                    # Выборка для скорости
                    X = df_scaled[features]
                    sample_size = min(3000, len(X))
                    if len(X) > sample_size:
                        X_sample = X.sample(sample_size, random_state=42)
                        sample_indices = X_sample.index
                    else:
                        X_sample = X
                        sample_indices = X.index
                    
                    with st.spinner(f"Выполняется кластеризация ({algorithm})..."):
                        if algorithm == "K-Means":
                            model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                            clusters = model.fit_predict(X_sample)
                            
                            # Метрики
                            inertia = model.inertia_
                            silhouette = silhouette_score(X_sample, clusters)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Inertia", f"{inertia:.2f}")
                            with col2:
                                st.metric("Silhouette Score", f"{silhouette:.3f}")
                            
                            # Для K-Means количество кластеров известно
                            n_clusters_found = n_clusters
                            noise_points = 0
                            
                        else:
                            model = DBSCAN(eps=eps, min_samples=min_samples)
                            clusters = model.fit_predict(X_sample)
                            
                            n_clusters_found = len(set(clusters)) - (1 if -1 in clusters else 0)
                            noise_points = np.sum(clusters == -1)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Найдено кластеров", n_clusters_found)
                            with col2:
                                st.metric("Шумовых точек", noise_points)
                    
                    # Визуализация
                    df_viz = daily_df.loc[sample_indices].copy()
                    df_viz['Cluster'] = clusters
                    
                    fig = px.scatter(
                        df_viz,
                        x=features[0],
                        y=features[1],
                        color='Cluster',
                        title=f"Кластеризация: {features[0]} vs {features[1]}",
                        hover_data=['city_name'] if 'city_name' in df_viz.columns else None
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Анализ кластеров (используем общую переменную n_clusters_found)
                    if 'Cluster' in df_viz.columns and n_clusters_found > 0:
                        st.subheader("Статистика по кластерам")
                        cluster_stats = df_viz.groupby('Cluster')[features].mean()
                        st.dataframe(cluster_stats.style.background_gradient(cmap='coolwarm'), use_container_width=True)
            
            # ========== РЕГРЕССИЯ ==========
            elif analysis_method == "Регрессия":
                st.header("Регрессионный анализ")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    target = st.selectbox("Целевая переменная (Y):", numeric_cols)
                
                with col2:
                    available_features = [col for col in numeric_cols if col != target]
                    features = st.multiselect(
                        "Признаки (X):",
                        available_features,
                        default=available_features[:3] if len(available_features) >= 3 else available_features
                    )
                
                if target and features:
                    model_type = st.selectbox(
                        "Тип модели:",
                        ["Linear Regression", "Ridge", "Lasso", "Random Forest"]
                    )
                    
                    test_size = st.slider("Тестовая выборка (%):", 10, 40, 20)
                    
                    # Подготовка данных
                    X = df_scaled[features]
                    y = df_scaled[target]
                    
                    # Выборка для скорости
                    sample_size = min(5000, len(X))
                    if len(X) > sample_size:
                        X_sample = X.sample(sample_size, random_state=42)
                        y_sample = y.loc[X_sample.index]
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_sample, y_sample, test_size=test_size/100, random_state=42
                        )
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size/100, random_state=42
                        )
                    
                    with st.spinner(f"Обучение модели {model_type}..."):
                        if model_type == "Linear Regression":
                            model = LinearRegression()
                        elif model_type == "Ridge":
                            alpha = st.slider("Alpha:", 0.01, 10.0, 1.0)
                            model = Ridge(alpha=alpha)
                        elif model_type == "Lasso":
                            alpha = st.slider("Alpha:", 0.01, 10.0, 1.0)
                            model = Lasso(alpha=alpha)
                        else:
                            n_estimators = st.slider("Количество деревьев:", 10, 100, 50)
                            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                        
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
                        st.metric("R² Train", f"{r2_train:.3f}")
                    with col2:
                        st.metric("R² Test", f"{r2_test:.3f}")
                    with col3:
                        st.metric("MAE Train", f"{mae_train:.3f}")
                    with col4:
                        st.metric("MAE Test", f"{mae_test:.3f}")
                    
                    # Визуализация
                    tab1, tab2 = st.tabs(["Прогнозы", "Важность признаков"])
                    
                    with tab1:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=y_test,
                            y=y_pred_test,
                            mode='markers',
                            name='Test данные'
                        ))
                        
                        min_val = min(y_test.min(), y_pred_test.min())
                        max_val = max(y_test.max(), y_pred_test.max())
                        fig.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Идеальный прогноз',
                            line=dict(dash='dash')
                        ))
                        
                        fig.update_layout(
                            title="Фактические vs Предсказанные значения",
                            xaxis_title="Фактические значения",
                            yaxis_title="Предсказанные значения"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        if hasattr(model, 'feature_importances_'):
                            importance = pd.DataFrame({
                                'Признак': features,
                                'Важность': model.feature_importances_
                            }).sort_values('Важность', ascending=False)
                            
                            fig = px.bar(importance, x='Признак', y='Важность', title="Важность признаков")
                            st.plotly_chart(fig, use_container_width=True)
                        elif hasattr(model, 'coef_'):
                            coefs = pd.DataFrame({
                                'Признак': features,
                                'Коэффициент': model.coef_
                            }).sort_values('Коэффициент', ascending=False)
                            
                            fig = px.bar(coefs, x='Признак', y='Коэффициент', title="Коэффициенты модели")
                            st.plotly_chart(fig, use_container_width=True)
            
            # ========== ВРЕМЕННЫЕ РЯДЫ ==========
            elif analysis_method == "Временные ряды":
                st.header("Анализ временных рядов")
                
                if 'date' not in daily_df.columns:
                    st.warning("Для анализа временных рядов нужна колонка с датами")
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'city_name' in daily_df.columns:
                            city = st.selectbox(
                                "Выберите город:",
                                ['Все'] + daily_df['city_name'].unique().tolist()[:10]
                            )
                            if city != 'Все':
                                df_city = daily_df[daily_df['city_name'] == city]
                            else:
                                df_city = daily_df
                        else:
                            df_city = daily_df
                    
                    with col2:
                        variable = st.selectbox("Выберите переменную:", numeric_cols)
                    
                    # Агрегация по дате
                    df_ts = df_city.groupby('date')[variable].mean().reset_index()
                    df_ts = df_ts.sort_values('date')
                    
                    # Визуализация
                    fig = px.line(df_ts, x='date', y=variable, title=f"{variable} по времени")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Простой анализ тренда
                    window = st.slider("Окно для скользящего среднего:", 7, 90, 30)
                    df_ts['moving_avg'] = df_ts[variable].rolling(window=window, center=True).mean()
                    
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=df_ts['date'], y=df_ts[variable],
                        name='Исходные данные', mode='lines'
                    ))
                    fig_trend.add_trace(go.Scatter(
                        x=df_ts['date'], y=df_ts['moving_avg'],
                        name=f'Скользящее среднее ({window} дней)',
                        line=dict(width=3)
                    ))
                    
                    fig_trend.update_layout(
                        title=f"Тренд {variable}",
                        xaxis_title="Дата",
                        yaxis_title=variable
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
            
            # ========== PCA ==========
            else:
                st.header("Анализ главных компонент (PCA)")
                
                pca_features = st.multiselect(
                    "Выберите признаки для PCA:",
                    numeric_cols,
                    default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
                )
                
                if len(pca_features) >= 2:
                    n_components = st.slider(
                        "Количество компонент:",
                        2, min(5, len(pca_features)), 3
                    )
                    
                    # Выборка для скорости
                    X_pca = df_scaled[pca_features]
                    sample_size = min(3000, len(X_pca))
                    if len(X_pca) > sample_size:
                        X_pca_sample = X_pca.sample(sample_size, random_state=42)
                    else:
                        X_pca_sample = X_pca
                    
                    with st.spinner("Выполняется PCA анализ..."):
                        pca = PCA(n_components=n_components)
                        X_pca_transformed = pca.fit_transform(X_pca_sample)
                        
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
                        title="Объясненная дисперсия",
                        yaxis=dict(title='Доля дисперсии'),
                        yaxis2=dict(
                            title='Накопленная дисперсия',
                            overlaying='y',
                            side='right'
                        )
                    )
                    st.plotly_chart(fig_var, use_container_width=True)
                    
                    # 2D визуализация
                    if n_components >= 2:
                        df_pca_viz = daily_df.loc[X_pca_sample.index].copy()
                        df_pca_viz['PC1'] = X_pca_transformed[:, 0]
                        df_pca_viz['PC2'] = X_pca_transformed[:, 1]
                        
                        # Упрощенная опция цвета - можно убрать или оставить только по числовым признакам
                        color_options = ['Нет'] + pca_features[:3]
                        color_by = st.selectbox("Цвет по:", color_options)
                        
                        fig_pca = px.scatter(
                            df_pca_viz,
                            x='PC1',
                            y='PC2',
                            color=color_by if color_by != 'Нет' else None,
                            title="PCA - Первые две компоненты"
                        )
                        st.plotly_chart(fig_pca, use_container_width=True)

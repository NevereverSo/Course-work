import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML библиотеки
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (r2_score, mean_absolute_error, 
                           mean_absolute_percentage_error, mean_squared_error)
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
    
    id_columns = ['station_id', 'id', 'station', 'station_code', 'station_number']
    
    filtered_cols = []
    for col in numeric_cols:
        is_id_column = False
        for id_pattern in id_columns:
            if id_pattern in col.lower():
                is_id_column = True
                break
        
        if not is_id_column and col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.95:
                filtered_cols.append(col)
    
    return filtered_cols

@st.cache_data
def prepare_scaled_data(_df, numeric_cols):
    """Подготовка масштабированных данных с кэшированием"""
    scaler = StandardScaler()
    df_scaled = _df.copy()
    
    df_filled = _df[numeric_cols].fillna(_df[numeric_cols].mean())
    df_scaled[numeric_cols] = scaler.fit_transform(df_filled)
    
    return df_scaled

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
st.sidebar.title("Weather Analytics")

page = st.sidebar.radio(
    "Навигация",
    ["Визуализация данных", "Анализ данных"]
)

if not daily_df.empty:
    st.sidebar.success(f"Данные загружены: {len(daily_df):,} записей")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Информация о данных")
    
    numeric_cols = get_numeric_columns(daily_df)
    st.sidebar.info(f"Числовых признаков: {len(numeric_cols)}")
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
        st.error("Данные не загружены.")
    else:
        numeric_cols = get_numeric_columns(daily_df)
        
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
        
        # Проверка нормальности распределения
        st.subheader("Проверка нормальности распределения")
        
        if numeric_cols:
            test_col = st.selectbox("Выберите признак для проверки нормальности:", numeric_cols)
            
            if test_col in daily_df.columns:
                data = daily_df[test_col].dropna()
                
                if len(data) > 0:
                    # Тест Шапиро-Уилка
                    shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(5000, len(data))))
                    
                    # Тест Колмогорова-Смирнова
                    ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                    
                    # Q-Q plot
                    fig = go.Figure()
                    
                    # Теоретические квантили
                    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
                    sample_quantiles = np.sort(data)
                    
                    fig.add_trace(go.Scatter(
                        x=theoretical_quantiles,
                        y=sample_quantiles,
                        mode='markers',
                        name='Данные'
                    ))
                    
                    # Линия идеального нормального распределения
                    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
                    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Идеальное нормальное распределение',
                        line=dict(dash='dash', color='red')
                    ))
                    
                    fig.update_layout(
                        title=f"Q-Q Plot для {test_col}",
                        xaxis_title="Теоретические квантили",
                        yaxis_title="Выборочные квантили"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Результаты тестов
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Тест Шапиро-Уилка", 
                                 f"p-value: {shapiro_p:.4f}",
                                 delta="Нормальное" if shapiro_p > 0.05 else "Не нормальное",
                                 delta_color="normal" if shapiro_p > 0.05 else "inverse")
                    
                    with col2:
                        st.metric("Тест Колмогорова-Смирнова",
                                 f"p-value: {ks_p:.4f}",
                                 delta="Нормальное" if ks_p > 0.05 else "Не нормальное",
                                 delta_color="normal" if ks_p > 0.05 else "inverse")
                    
                    # Гистограмма с нормальной кривой
                    fig_hist = px.histogram(
                        daily_df, 
                        x=test_col, 
                        nbins=50,
                        title=f"Распределение {test_col} с нормальной кривой",
                        marginal="box"
                    )
                    
                    # Добавляем нормальную кривую
                    x_range = np.linspace(data.min(), data.max(), 100)
                    pdf = stats.norm.pdf(x_range, data.mean(), data.std())
                    fig_hist.add_trace(go.Scatter(
                        x=x_range,
                        y=pdf * len(data) * (data.max() - data.min()) / 50,
                        mode='lines',
                        name='Нормальное распределение',
                        line=dict(color='red', width=2)
                    ))
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
        
        tab1, tab2, tab3 = st.tabs(["Данные", "Распределения", "Корреляции"])
        
        with tab1:
            st.header("Просмотр данных")
            
            dataset_choice = st.selectbox(
                "Выберите датасет:",
                ["Ежедневные данные", "Города", "Страны"]
            )
            
            if dataset_choice == "Ежедневные данные":
                df_display = daily_df
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
                preview_rows = st.slider("Показать строк:", 100, 1000, 500, step=100)
                st.dataframe(df_display.head(preview_rows), use_container_width=True)
        
        with tab2:
            st.header("Анализ распределений")
            
            if numeric_cols:
                selected_col = st.selectbox("Выберите признак:", numeric_cols)
                plot_type = st.radio(
                    "Тип графика:",
                    ["Гистограмма", "Box Plot", "Violin Plot"]
                )
                
                if plot_type == "Гистограмма":
                    fig = px.histogram(daily_df, x=selected_col, nbins=50)
                elif plot_type == "Box Plot":
                    fig = px.box(daily_df, y=selected_col)
                else:
                    fig = px.violin(daily_df, y=selected_col)
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("Корреляционный анализ")
            
            if len(numeric_cols) > 1:
                max_features = min(10, len(numeric_cols))
                selected_features = st.multiselect(
                    "Выберите признаки для анализа:",
                    numeric_cols,
                    default=numeric_cols[:max_features]
                )
                
                if len(selected_features) > 1:
                    corr_matrix = daily_df[selected_features].corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=".2f",
                        aspect="auto",
                        title="Корреляционная матрица",
                        color_continuous_scale="RdBu_r"
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# PAGE 2 — АНАЛИЗ ДАННЫХ
# ==========================================================
else:
    
    if daily_df.empty:
        st.error("Для анализа нужны ежедневные данные")
    else:
        numeric_cols = get_numeric_columns(daily_df)
        
        if not numeric_cols:
            st.error("Нет числовых признаков для анализа.")
        else:
            st.info(f"Доступно для анализа: {len(numeric_cols)} числовых признаков")
            
            analysis_method = st.selectbox(
                "Метод анализа:",
                ["Кластеризация", "Регрессия", "Временные ряды", "PCA"]
            )
            
            df_scaled = prepare_scaled_data(daily_df, numeric_cols)
            
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
                            inertia = model.inertia_
                            silhouette = silhouette_score(X_sample, clusters)
                            n_clusters_found = n_clusters
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Inertia", f"{inertia:.2f}")
                            with col2:
                                st.metric("Silhouette Score", f"{silhouette:.3f}")
                            
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
                    
                    df_viz = daily_df.loc[sample_indices].copy()
                    df_viz['Cluster'] = clusters
                    
                    fig = px.scatter(
                        df_viz,
                        x=features[0],
                        y=features[1],
                        color='Cluster',
                        title=f"Кластеризация: {features[0]} vs {features[1]}"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
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
                    test_size = st.slider("Тестовая выборка (%):", 10, 40, 20)
                    
                    X = df_scaled[features]
                    y = df_scaled[target]
                    
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
                    
                    # Словарь для хранения результатов
                    results = {}
                    
                    # Модели для сравнения
                    models = {
                        "Линейная регрессия": LinearRegression(),
                        "Гребневая регрессия (Ridge)": Ridge(alpha=1.0),
                        "Лассо регрессия (Lasso)": Lasso(alpha=0.1),
                        "Случайный лес (Random Forest)": RandomForestRegressor(n_estimators=100, random_state=42),
                        "Градиентный бустинг (Gradient Boosting)": GradientBoostingRegressor(n_estimators=100, random_state=42)
                    }
                    
                    # Настройки гиперпараметров
                    model_params = {}
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        ridge_alpha = st.slider("Alpha для Ridge:", 0.01, 10.0, 1.0, key="ridge_alpha")
                        models["Гребневая регрессия (Ridge)"] = Ridge(alpha=ridge_alpha)
                        
                        lasso_alpha = st.slider("Alpha для Lasso:", 0.001, 1.0, 0.1, key="lasso_alpha")
                        models["Лассо регрессия (Lasso)"] = Lasso(alpha=lasso_alpha)
                    
                    with col2:
                        rf_estimators = st.slider("Деревья для Random Forest:", 10, 200, 100, key="rf_estimators")
                        models["Случайный лес (Random Forest)"] = RandomForestRegressor(
                            n_estimators=rf_estimators, random_state=42
                        )
                        
                        gb_estimators = st.slider("Деревья для Gradient Boosting:", 10, 200, 100, key="gb_estimators")
                        models["Градиентный бустинг (Gradient Boosting)"] = GradientBoostingRegressor(
                            n_estimators=gb_estimators, random_state=42
                        )
                    
                    # Обучение и оценка моделей
                    progress_bar = st.progress(0)
                    for idx, (name, model) in enumerate(models.items()):
                        with st.spinner(f"Обучение {name}..."):
                            model.fit(X_train, y_train)
                            y_pred_train = model.predict(X_train)
                            y_pred_test = model.predict(X_test)
                            
                            # Вычисление метрик
                            r2_train = r2_score(y_train, y_pred_train)
                            r2_test = r2_score(y_test, y_pred_test)
                            mae_train = mean_absolute_error(y_train, y_pred_train)
                            mae_test = mean_absolute_error(y_test, y_pred_test)
                            mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
                            mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
                            mse_train = mean_squared_error(y_train, y_pred_train)
                            mse_test = mean_squared_error(y_test, y_pred_test)
                            rmse_train = np.sqrt(mse_train)
                            rmse_test = np.sqrt(mse_test)
                            
                            results[name] = {
                                'R² Train': r2_train,
                                'R² Test': r2_test,
                                'MAE Train': mae_train,
                                'MAE Test': mae_test,
                                'MAPE Train': mape_train,
                                'MAPE Test': mape_test,
                                'RMSE Train': rmse_train,
                                'RMSE Test': rmse_test
                            }
                        
                        progress_bar.progress((idx + 1) / len(models))
                    
                    # Сравнительная таблица
                    st.subheader("Сравнительная таблица моделей")
                    
                    results_df = pd.DataFrame(results).T
                    results_df = results_df.round(4)
                    
                    # Отображение таблицы
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Визуализация сравнения моделей
                    st.subheader("Визуализация сравнения моделей")
                    
                    metric_to_plot = st.selectbox(
                        "Выберите метрику для сравнения:",
                        ['R² Test', 'MAE Test', 'MAPE Test', 'RMSE Test']
                    )
                    
                    fig_comparison = go.Figure()
                    
                    for model_name in results.keys():
                        fig_comparison.add_trace(go.Bar(
                            x=[model_name],
                            y=[results[model_name][metric_to_plot]],
                            name=model_name,
                            text=[f"{results[model_name][metric_to_plot]:.4f}"],
                            textposition='auto'
                        ))
                    
                    fig_comparison.update_layout(
                        title=f"Сравнение моделей по метрике: {metric_to_plot}",
                        xaxis_title="Модель",
                        yaxis_title=metric_to_plot,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Детальный анализ лучшей модели
                    st.subheader("Детальный анализ лучшей модели")
                    
                    # Находим лучшую модель по R² Test
                    best_model_name = max(results.keys(), key=lambda x: results[x]['R² Test'])
                    best_model = models[best_model_name]
                    
                    st.write(f"**Лучшая модель:** {best_model_name}")
                    st.write(f"**R² Test:** {results[best_model_name]['R² Test']:.4f}")
                    st.write(f"**MAE Test:** {results[best_model_name]['MAE Test']:.4f}")
                    
                    # График фактических vs предсказанных значений для лучшей модели
                    best_y_pred_test = best_model.predict(X_test)
                    
                    fig_best = go.Figure()
                    fig_best.add_trace(go.Scatter(
                        x=y_test,
                        y=best_y_pred_test,
                        mode='markers',
                        name='Test данные'
                    ))
                    
                    min_val = min(y_test.min(), best_y_pred_test.min())
                    max_val = max(y_test.max(), best_y_pred_test.max())
                    fig_best.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Идеальный прогноз',
                        line=dict(dash='dash')
                    ))
                    
                    fig_best.update_layout(
                        title=f"Фактические vs Предсказанные значения ({best_model_name})",
                        xaxis_title="Фактические значения",
                        yaxis_title="Предсказанные значения"
                    )
                    st.plotly_chart(fig_best, use_container_width=True)
                    
                    # Важность признаков (если модель поддерживает)
                    if hasattr(best_model, 'feature_importances_'):
                        importance = pd.DataFrame({
                            'Признак': features,
                            'Важность': best_model.feature_importances_
                        }).sort_values('Важность', ascending=False)
                        
                        fig_importance = px.bar(
                            importance, 
                            x='Признак', 
                            y='Важность',
                            title=f"Важность признаков ({best_model_name})"
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                    
                    elif hasattr(best_model, 'coef_'):
                        coefs = pd.DataFrame({
                            'Признак': features,
                            'Коэффициент': best_model.coef_
                        }).sort_values('Коэффициент', ascending=False)
                        
                        fig_coef = px.bar(
                            coefs,
                            x='Признак',
                            y='Коэффициент',
                            title=f"Коэффициенты модели ({best_model_name})"
                        )
                        st.plotly_chart(fig_coef, use_container_width=True)
            
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
                    
                    df_ts = df_city.groupby('date')[variable].mean().reset_index()
                    df_ts = df_ts.sort_values('date')
                    
                    fig = px.line(df_ts, x='date', y=variable)
                    st.plotly_chart(fig, use_container_width=True)
            
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

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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (silhouette_score, r2_score, mean_absolute_error, 
                           mean_absolute_percentage_error, mean_squared_error)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Weather Analytics Dashboard", 
    layout="wide"
)

# ----------------------------------------------------------
# LOAD DATA С ОПТИМИЗАЦИЕЙ (без обработки пропусков и дубликатов)
# ----------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Загрузка данных...")
def load_data():
    """Быстрая загрузка данных (предполагаем, что данные уже очищены)"""
    try:
        daily_df = pd.read_csv("daily_weather_smallest.csv", low_memory=False)
        if not daily_df.empty and 'date' in daily_df.columns:
            daily_df['date'] = pd.to_datetime(daily_df['date'], errors='coerce')
    except Exception as e:
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
countries_df, cities_df, daily_df = load_data()

# ----------------------------------------------------------
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ С КЭШЕМ
# ----------------------------------------------------------
@st.cache_data
def get_numeric_columns(df):
    """Быстрое получение числовых колонок без station_id"""
    if df.empty:
        return []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Быстро фильтруем ID колонки
    id_keywords = ['station_id', 'id', 'station', '_id']
    filtered_cols = [col for col in numeric_cols 
                     if not any(keyword in col.lower() for keyword in id_keywords)]
    
    return filtered_cols

@st.cache_data
def prepare_scaled_data(_df, numeric_cols):
    """Быстрая подготовка масштабированных данных (без обработки пропусков)"""
    if len(_df) == 0 or len(numeric_cols) == 0:
        return pd.DataFrame()
    
    scaler = StandardScaler()
    
    # Используем выборку для скорости
    sample_size = min(3000, len(_df))
    if len(_df) > sample_size:
        df_sample = _df.sample(sample_size, random_state=42)
        scaler.fit(df_sample[numeric_cols])
    else:
        scaler.fit(_df[numeric_cols])
    
    # Применяем трансформацию ко всем данным
    df_scaled = _df.copy()
    df_scaled[numeric_cols] = scaler.transform(_df[numeric_cols])
    
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
    numeric_cols = get_numeric_columns(daily_df)
    st.sidebar.success(f"Данные: {len(daily_df):,} записей, {len(numeric_cols)} признаков")
else:
    st.sidebar.error("Данные не загружены")

# ==========================================================
# PAGE 1 — ВИЗУАЛИЗАЦИЯ ДАННЫХ (ОПТИМИЗИРОВАННАЯ)
# ==========================================================
if page == "Визуализация данных":
    
    if daily_df.empty:
        st.error("Данные не загружены.")
    else:
        numeric_cols = get_numeric_columns(daily_df)
        
        # Быстрые KPI
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'city_name' in daily_df.columns:
                st.metric("Городов", daily_df['city_name'].nunique())
            else:
                st.metric("Записей", len(daily_df))
        
        with col2:
            if 'date' in daily_df.columns:
                st.metric("Дней данных", len(daily_df['date'].dt.date.unique()))
        
        with col3:
            st.metric("Признаков", len(numeric_cols))
        
        # Быстрый анализ распределения
        if numeric_cols:
            st.subheader("Быстрый анализ признаков")
            
            # Выбор признака через selectbox (быстрее чем multiselect)
            selected_col = st.selectbox(
                "Выберите признак для анализа:", 
                numeric_cols[:15]  # Ограничиваем выбор для selectbox
            )
            
            if selected_col in daily_df.columns:
                # Простая статистика
                data = daily_df[selected_col]
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Среднее", f"{data.mean():.2f}")
                with col2:
                    st.metric("Медиана", f"{data.median():.2f}")
                with col3:
                    st.metric("Стд. отклонение", f"{data.std():.2f}")
                with col4:
                    st.metric("Диапазон", f"{data.min():.1f}-{data.max():.1f}")
                
                # Быстрая визуализация
                fig = px.histogram(
                    daily_df, 
                    x=selected_col, 
                    nbins=30,
                    title=f"Распределение {selected_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Вкладки с минимальной функциональностью
        tab1, tab2 = st.tabs(["Данные", "Корреляции"])
        
        with tab1:
            st.header("Просмотр данных")
            
            # Быстрый выбор датасета
            dataset_choice = st.radio(
                "Датасет:",
                ["Ежедневные данные", "Города", "Страны"],
                horizontal=True
            )
            
            if dataset_choice == "Ежедневные данные":
                df_display = daily_df
                
                # Быстрый фильтр по городу (если есть)
                if 'city_name' in daily_df.columns:
                    cities = daily_df['city_name'].unique()[:3]  # Только 3 города для скорости
                    selected_city = st.selectbox(
                        "Город:", 
                        ['Все'] + list(cities)
                    )
                    if selected_city != 'Все':
                        df_display = df_display[df_display['city_name'] == selected_city]
            
            elif dataset_choice == "Города" and not cities_df.empty:
                df_display = cities_df
            elif dataset_choice == "Страны" and not countries_df.empty:
                df_display = countries_df
            else:
                df_display = pd.DataFrame()
            
            if not df_display.empty:
                # Быстрый предпросмотр
                st.dataframe(df_display.head(100), use_container_width=True)
        
        with tab2:
            st.header("Корреляционный анализ")
            
            if len(numeric_cols) > 1:
                # Убрано ограничение на количество признаков - пользователь выбирает сколько хочет
                selected_features = st.multiselect(
                    "Выберите признаки для анализа корреляций:",
                    numeric_cols,
                    default=numeric_cols[:min(10, len(numeric_cols))]  # По умолчанию первые 10 или меньше
                )
                
                if len(selected_features) > 1:
                    st.write(f"**Анализ корреляций между {len(selected_features)} признаками:**")
                    
                    # Вычисляем корреляции - используем все данные, так как это быстро
                    corr_matrix = daily_df[selected_features].corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=".2f",
                        aspect="auto",
                        title=f"Корреляционная матрица ({len(selected_features)} признаков)",
                        color_continuous_scale="RdBu_r"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Показываем топ-5 корреляций
                    st.subheader("Наиболее сильные корреляции")
                    corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr = corr_matrix.iloc[i, j]
                            corr_pairs.append({
                                'Признак 1': corr_matrix.columns[i],
                                'Признак 2': corr_matrix.columns[j],
                                'Корреляция': corr
                            })
                    
                    if corr_pairs:
                        corr_df = pd.DataFrame(corr_pairs)
                        corr_df['abs_corr'] = corr_df['Корреляция'].abs()
                        
                        # Показываем топ-5 по абсолютному значению
                        top_n = min(5, len(corr_df))
                        top_correlations = corr_df.nlargest(top_n, 'abs_corr')
                        
                        # Отображаем в виде таблицы
                        display_df = top_correlations[['Признак 1', 'Признак 2', 'Корреляция']].copy()
                        display_df['Корреляция'] = display_df['Корреляция'].apply(lambda x: f"{x:.3f}")
                        
                        st.dataframe(
                            display_df,
                            column_config={
                                "Признак 1": "Первый признак",
                                "Признак 2": "Второй признак", 
                                "Корреляция": "Коэффициент корреляции"
                            },
                            use_container_width=True
                        )
                    
                    # Дополнительно: scatter plot для наиболее коррелированной пары
                    if len(selected_features) >= 2 and len(corr_pairs) > 0:
                        st.subheader("Визуализация наиболее коррелированной пары")
                        
                        # Находим пару с максимальной абсолютной корреляцией
                        strongest_idx = corr_df['abs_corr'].idxmax()
                        strongest_pair = corr_df.loc[strongest_idx]
                        
                        # Упрощенный scatter plot без trendline="ols"
                        fig_scatter = px.scatter(
                            daily_df,
                            x=strongest_pair['Признак 1'],
                            y=strongest_pair['Признак 2'],
                            title=f"{strongest_pair['Признак 2']} vs {strongest_pair['Признак 1']} (r = {strongest_pair['Корреляция']:.3f})"
                        )
                        
                        # Вместо trendline="ols" добавляем свою линию регрессии
                        # Простая линейная регрессия через numpy
                        x_data = daily_df[strongest_pair['Признак 1']].values
                        y_data = daily_df[strongest_pair['Признак 2']].values
                        
                        # Удаляем NaN значения
                        mask = ~np.isnan(x_data) & ~np.isnan(y_data)
                        x_clean = x_data[mask]
                        y_clean = y_data[mask]
                        
                        if len(x_clean) > 1:
                            # Вычисляем коэффициенты линейной регрессии
                            coeffs = np.polyfit(x_clean, y_clean, 1)
                            
                            # Создаем линию регрессии
                            x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
                            y_pred = coeffs[0] * x_range + coeffs[1]
                            
                            fig_scatter.add_trace(go.Scatter(
                                x=x_range,
                                y=y_pred,
                                mode='lines',
                                name='Линия регрессии',
                                line=dict(color='red', width=2)
                            ))
                        
                        st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Выберите как минимум 2 признака для анализа корреляций.")

# ==========================================================
# PAGE 2 — АНАЛИЗ ДАННЫХ (ОПТИМИЗИРОВАННЫЙ)
# ==========================================================
else:
    
    if daily_df.empty:
        st.error("Для анализа нужны данные.")
    else:
        numeric_cols = get_numeric_columns(daily_df)
        
        if not numeric_cols:
            st.error("Нет числовых признаков для анализа.")
        else:
            st.write(f"**Доступно признаков:** {len(numeric_cols)}")
            
            analysis_method = st.selectbox(
                "Метод анализа:",
                ["Регрессия", "Кластеризация", "PCA"],
                index=0  # Регрессия по умолчанию
            )
            
            # Готовим данные только когда нужно
            if analysis_method in ["Регрессия", "Кластеризация", "PCA"]:
                df_scaled = prepare_scaled_data(daily_df, numeric_cols)
            
            if analysis_method == "Регрессия":
                st.header("Регрессионный анализ")
                
                # Быстрый выбор целевой переменной
                target = st.selectbox(
                    "Целевая переменная (Y):", 
                    numeric_cols[:10]  # Ограничиваем выбор
                )
                
                if target:
                    # Автоматический выбор признаков (топ-3 по корреляции)
                    if len(numeric_cols) > 1:
                        # Вычисляем корреляции с целевой переменной
                        correlations = daily_df[numeric_cols].corr()[target].abs().sort_values(ascending=False)
                        # Исключаем саму целевую переменную
                        correlations = correlations[correlations.index != target]
                        # Берем топ-3 наиболее коррелированных признака
                        top_features = correlations.head(3).index.tolist()
                    else:
                        top_features = []
                    
                    features = st.multiselect(
                        "Признаки (X):",
                        numeric_cols,
                        default=top_features
                    )
                
                if target and features:
                    test_size = st.slider("Тестовая выборка:", 0.1, 0.4, 0.2, 0.05)
                    
                    # Готовим данные
                    X = df_scaled[features]
                    y = df_scaled[target]
                    
                    # Используем выборку для скорости
                    sample_size = min(2000, len(X))
                    if len(X) > sample_size:
                        sample_idx = np.random.choice(len(X), sample_size, replace=False)
                        X_sample = X.iloc[sample_idx]
                        y_sample = y.iloc[sample_idx]
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_sample, y_sample, test_size=test_size, random_state=42
                        )
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                    
                    # Модели для сравнения
                    models_config = {
                        "Линейная регрессия": {
                            "model": LinearRegression(),
                            "params": {}
                        },
                        "Гребневая регрессия": {
                            "model": Ridge(),
                            "params": {"alpha": 1.0}
                        },
                        "Лассо регрессия": {
                            "model": Lasso(),
                            "params": {"alpha": 0.01}
                        },
                        "Random Forest": {
                            "model": RandomForestRegressor(),
                            "params": {"n_estimators": 50, "random_state": 42}
                        },
                        "Gradient Boosting": {
                            "model": GradientBoostingRegressor(),
                            "params": {"n_estimators": 50, "random_state": 42}
                        }
                    }
                    
                    # Настраиваем параметры
                    col1, col2 = st.columns(2)
                    with col1:
                        ridge_alpha = st.slider("Alpha для Ridge:", 0.01, 10.0, 1.0, 0.01)
                        models_config["Гребневая регрессия"]["params"]["alpha"] = ridge_alpha
                        
                        lasso_alpha = st.slider("Alpha для Lasso:", 0.001, 1.0, 0.01, 0.001)
                        models_config["Лассо регрессия"]["params"]["alpha"] = lasso_alpha
                    
                    # Обучение моделей
                    results = {}
                    progress_bar = st.progress(0)
                    
                    for idx, (name, config) in enumerate(models_config.items()):
                        model = config["model"]
                        model.set_params(**config["params"])
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        # Метрики
                        results[name] = {
                            'R²': r2_score(y_test, y_pred),
                            'MAE': mean_absolute_error(y_test, y_pred),
                            'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100,  # в процентах
                            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
                        }
                        
                        progress_bar.progress((idx + 1) / len(models_config))
                    
                    # Сравнительная таблица
                    st.subheader("Сравнение моделей")
                    
                    results_df = pd.DataFrame(results).T.round(4)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Визуализация лучшей модели
                    best_model_name = max(results.keys(), key=lambda x: results[x]['R²'])
                    best_config = models_config[best_model_name]
                    best_model = best_config["model"]
                    best_model.set_params(**best_config["params"])
                    best_model.fit(X_train, y_train)
                    
                    st.subheader(f"Лучшая модель: {best_model_name}")
                    
                    # График прогнозов
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=y_test,
                        y=best_model.predict(X_test),
                        mode='markers',
                        name='Прогнозы'
                    ))
                    
                    # Линия идеального прогноза
                    min_val = min(y_test.min(), best_model.predict(X_test).min())
                    max_val = max(y_test.max(), best_model.predict(X_test).max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Идеально',
                        line=dict(dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="Фактические vs Предсказанные значения",
                        xaxis_title="Фактические значения",
                        yaxis_title="Предсказанные значения"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_method == "Кластеризация":
                st.header("Кластеризация")
                
                # Быстрый выбор признаков
                if len(numeric_cols) >= 2:
                    default_features = numeric_cols[:2]
                else:
                    default_features = numeric_cols
                
                features = st.multiselect(
                    "Признаки для кластеризации:",
                    numeric_cols[:6],  # Ограничиваем выбор
                    default=default_features
                )
                
                if len(features) >= 2:
                    algorithm = st.selectbox("Алгоритм:", ["K-Means", "DBSCAN"])
                    
                    if algorithm == "K-Means":
                        n_clusters = st.slider("Кластеров:", 2, 6, 3)
                    else:
                        eps = st.slider("EPS:", 0.1, 1.0, 0.5, 0.1)
                    
                    # Готовим данные
                    X = df_scaled[features]
                    
                    # Используем выборку
                    sample_size = min(1000, len(X))
                    if len(X) > sample_size:
                        X_sample = X.sample(sample_size, random_state=42)
                    else:
                        X_sample = X
                    
                    # Кластеризация
                    if algorithm == "K-Means":
                        model = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
                        clusters = model.fit_predict(X_sample)
                        st.metric("Inertia", f"{model.inertia_:.2f}")
                    else:
                        model = DBSCAN(eps=eps, min_samples=5)
                        clusters = model.fit_predict(X_sample)
                    
                    # Визуализация
                    df_viz = daily_df.loc[X_sample.index].copy()
                    df_viz['Cluster'] = clusters
                    
                    fig = px.scatter(
                        df_viz,
                        x=features[0],
                        y=features[1],
                        color='Cluster',
                        title=f"Кластеризация: {features[0]} vs {features[1]}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            else:  # PCA
                st.header("Анализ главных компонент (PCA)")
                
                # Автоматический выбор признаков
                if len(numeric_cols) >= 4:
                    pca_features = numeric_cols[:4]
                else:
                    pca_features = numeric_cols
                
                if len(pca_features) >= 2:
                    n_components = min(3, len(pca_features))
                    
                    # PCA
                    X = df_scaled[pca_features]
                    
                    # Используем выборку
                    sample_size = min(1000, len(X))
                    if len(X) > sample_size:
                        X_sample = X.sample(sample_size, random_state=42)
                    else:
                        X_sample = X
                    
                    pca = PCA(n_components=n_components)
                    X_pca = pca.fit_transform(X_sample)
                    
                    # График объясненной дисперсии
                    explained_var = pca.explained_variance_ratio_
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[f"PC{i+1}" for i in range(n_components)],
                        y=explained_var,
                        name='Доля дисперсии'
                    ))
                    
                    fig.update_layout(
                        title="Объясненная дисперсия",
                        yaxis_title='Доля дисперсии'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 2D визуализация
                    if n_components >= 2:
                        df_viz = daily_df.loc[X_sample.index].copy()
                        df_viz['PC1'] = X_pca[:, 0]
                        df_viz['PC2'] = X_pca[:, 1]
                        
                        fig_scatter = px.scatter(
                            df_viz,
                            x='PC1',
                            y='PC2',
                            title="PCA - Проекция данных"
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

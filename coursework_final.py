import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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

# Для прогнозирования временных рядов
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(
    page_title="Weather Analytics Dashboard", 
    layout="wide"
)

# ----------------------------------------------------------
# LOAD DATA С ОПТИМИЗАЦИЕЙ
# ----------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Загрузка данных...")
def load_data():
    """Быстрая загрузка данных"""
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
# ФУНКЦИИ ДЛЯ ФИЛЬТРАЦИИ ПО ГОРОДАМ
# ----------------------------------------------------------
def get_available_cities(df):
    """Получить список доступных городов"""
    if df.empty or 'city_name' not in df.columns:
        return []
    cities = sorted(df['city_name'].dropna().unique().tolist())
    return ["Все города"] + cities

def filter_data_by_city(df, selected_city):
    """Фильтрация данных по выбранному городу"""
    if df.empty or not selected_city or selected_city == "Все города":
        return df.copy()
    
    return df[df['city_name'] == selected_city].copy()

def get_city_stats(df, city):
    """Получить статистику по городу"""
    if df.empty or city not in df['city_name'].values:
        return {}
    
    city_data = df[df['city_name'] == city]
    
    stats = {
        'total_days': len(city_data),
        'date_range': f"{city_data['date'].min().date()} - {city_data['date'].max().date()}",
        'avg_temp': round(city_data['avg_temp_c'].mean(), 2) if 'avg_temp_c' in city_data.columns else None,
        'avg_precipitation': round(city_data['precipitation_mm'].mean(), 2) if 'precipitation_mm' in city_data.columns else None,
        'avg_pressure': round(city_data['avg_sea_level_pres_hpa'].mean(), 2) if 'avg_sea_level_pres_hpa' in city_data.columns else None,
        'seasons': city_data['season'].unique().tolist() if 'season' in city_data.columns else []
    }
    
    return stats

# ----------------------------------------------------------
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ С КЭШЕМ
# ----------------------------------------------------------
@st.cache_data
def get_numeric_columns(df):
    """Быстрое получение числовых колонок"""
    if df.empty:
        return []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Фильтруем ID колонки
    id_keywords = ['station_id', 'id', 'station', '_id']
    filtered_cols = [col for col in numeric_cols 
                     if not any(keyword in col.lower() for keyword in id_keywords)]
    
    return filtered_cols

@st.cache_data
def prepare_scaled_data(_df, numeric_cols):
    """Быстрая подготовка масштабированных данных"""
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
    
    # Применяем трансформацию
    df_scaled = _df.copy()
    df_scaled[numeric_cols] = scaler.transform(_df[numeric_cols])
    
    return df_scaled

# ----------------------------------------------------------
# ФУНКЦИИ ДЛЯ ПРОГНОЗИРОВАНИЯ ВРЕМЕННЫХ РЯДОВ
# ----------------------------------------------------------
@st.cache_data(ttl=1800, max_entries=5)
def prepare_time_series_data(df, target_col, date_col='date'):
    """
    Подготовка данных временного ряда с оптимизацией
    """
    if df.empty or target_col not in df.columns or date_col not in df.columns:
        return None
    
    # Сортируем по дате и убираем дубликаты
    ts_df = df.sort_values(date_col).drop_duplicates(subset=[date_col])
    
    if len(ts_df) < 10:  # Минимум данных
        return None
    
    # Создаем временной ряд
    ts_data = ts_df[[date_col, target_col]].copy()
    ts_data.columns = ['ds', 'y']
    
    # Интерполяция пропусков
    ts_data['y'] = ts_data['y'].interpolate(method='linear')
    
    return ts_data

@st.cache_data(ttl=1800, max_entries=3)
def arima_forecast(ts_data, periods=30, order=(1,1,1)):
    """
    Прогнозирование ARIMA
    """
    try:
        # Используем последние 100 точек для скорости
        if len(ts_data) > 100:
            ts_series = ts_data.set_index('ds')['y'].iloc[-100:]
        else:
            ts_series = ts_data.set_index('ds')['y']
        
        model = ARIMA(ts_series, order=order)
        model_fit = model.fit()
        
        forecast = model_fit.forecast(steps=periods)
        last_date = ts_series.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        forecast_df = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecast.values
        })
        
        return model_fit, forecast_df
    except Exception as e:
        st.error(f"Ошибка ARIMA: {str(e)[:100]}")
        return None, None

@st.cache_data(ttl=1800, max_entries=3)
def exponential_smoothing_forecast(ts_data, periods=30):
    """
    Прогнозирование экспоненциальным сглаживанием
    """
    try:
        ts_series = ts_data.set_index('ds')['y']
        
        if len(ts_series) > 200:
            ts_series = ts_series.iloc[-200:]
        
        model = ExponentialSmoothing(
            ts_series,
            seasonal_periods=min(7, len(ts_series)),
            trend='add',
            seasonal='add'
        )
        model_fit = model.fit()
        
        forecast = model_fit.forecast(steps=periods)
        last_date = ts_series.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        forecast_df = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecast.values
        })
        
        return model_fit, forecast_df
    except Exception as e:
        st.error(f"Ошибка Exponential Smoothing: {str(e)[:100]}")
        return None, None

# ----------------------------------------------------------
# ФУНКЦИЯ ДЛЯ КОНВЕРТАЦИИ ДАТ В ЧИСЛОВОЙ ФОРМАТ
# ----------------------------------------------------------
def convert_dates_to_numeric(dates):
    """Конвертирует даты в числовой формат (количество дней с первой даты)"""
    if len(dates) == 0:
        return dates
    
    # Если это уже числовой формат, возвращаем как есть
    if np.issubdtype(dates.dtype, np.number):
        return dates
    
    # Конвертируем datetime в числовой формат
    if pd.api.types.is_datetime64_any_dtype(dates):
        # Используем разницу в днях от первой даты
        min_date = dates.min()
        numeric_dates = (dates - min_date).dt.days
        return numeric_dates
    elif hasattr(dates.iloc[0], 'date'):
        # Для объектов datetime.date
        min_date = min(dates)
        numeric_dates = [(date - min_date).days for date in dates]
        return pd.Series(numeric_dates)
    
    return dates

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
st.sidebar.title("Weather Analytics")

# Обновляем навигацию для добавления новой страницы
page = st.sidebar.radio(
    "Навигация",
    ["Визуализация данных", "Анализ данных", "Прогнозирование"]
)

# Выбор города в сайдбаре (доступен на всех страницах)
st.sidebar.subheader("Выбор города")

if not daily_df.empty and 'city_name' in daily_df.columns:
    available_cities = get_available_cities(daily_df)
    
    if available_cities:
        selected_city = st.sidebar.selectbox(
            "Выберите город для анализа:",
            options=available_cities,
            index=0,
            help="Выберите город для анализа. 'Все города' показывает сводную статистику."
        )
        
        # Фильтруем данные по выбранному городу
        filtered_df = filter_data_by_city(daily_df, selected_city)
        
        # Показываем статистику по выбранному городу
        if selected_city != "Все города":
            city_stats = get_city_stats(daily_df, selected_city)
            if city_stats:
                st.sidebar.info(f"""
                **Статистика {selected_city}:**
                - Дней данных: {city_stats['total_days']}
                - Период: {city_stats['date_range']}
                - Ср. температура: {city_stats['avg_temp']}°C
                - Ср. осадки: {city_stats['avg_precipitation']} мм
                - Ср. давление: {city_stats['avg_pressure']} гПа
                """)
        
        numeric_cols = get_numeric_columns(filtered_df)
        if selected_city == "Все города":
            st.sidebar.success(f"Все города: {len(filtered_df):,} записей, {len(filtered_df['city_name'].unique())} городов")
        else:
            st.sidebar.success(f"{selected_city}: {len(filtered_df):,} записей, {len(numeric_cols)} признаков")
    else:
        st.sidebar.warning("Колонка 'city_name' не найдена в данных")
        filtered_df = daily_df
        selected_city = "Все города"
        numeric_cols = get_numeric_columns(filtered_df)
else:
    st.sidebar.error("Данные не загружены или нет колонки с городами")
    filtered_df = daily_df
    selected_city = "Все города"
    numeric_cols = []

# ==========================================================
# PAGE 1 — ВИЗУАЛИЗАЦИЯ ДАННЫХ
# ==========================================================
if page == "Визуализация данных":
    
    if filtered_df.empty:
        st.error("Данные не загружены.")
    else:
        # Показываем информацию о выбранном городе
        if selected_city == "Все города":
            st.header(f"Визуализация данных: Все города")
        else:
            st.header(f"Визуализация данных: {selected_city}")
        
        numeric_cols = get_numeric_columns(filtered_df)
        
        # Быстрые KPI
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'city_name' in filtered_df.columns:
                if selected_city == "Все города":
                    unique_cities = filtered_df['city_name'].nunique()
                    st.metric("Городов", unique_cities)
                else:
                    st.metric("Выбран город", selected_city)
            else:
                st.metric("Записей", len(filtered_df))
        
        with col2:
            if 'date' in filtered_df.columns:
                unique_days = len(filtered_df['date'].dt.date.unique())
                st.metric("Дней данных", unique_days)
        
        with col3:
            st.metric("Признаков", len(numeric_cols))
        
        # Сводная статистика если выбраны "Все города"
        if selected_city == "Все города" and 'city_name' in filtered_df.columns and filtered_df['city_name'].nunique() > 1:
            with st.expander("Сводная статистика по городам"):
                city_stats_summary = []
                for city in filtered_df['city_name'].unique():
                    city_data = filtered_df[filtered_df['city_name'] == city]
                    stats = {
                        'Город': city,
                        'Дней': len(city_data),
                        'Ср. темп. (°C)': round(city_data['avg_temp_c'].mean(), 1) if 'avg_temp_c' in city_data.columns else 'N/A',
                        'Ср. осадки (мм)': round(city_data['precipitation_mm'].mean(), 1) if 'precipitation_mm' in city_data.columns else 'N/A',
                        'Ср. давление (гПа)': round(city_data['avg_sea_level_pres_hpa'].mean(), 1) if 'avg_sea_level_pres_hpa' in city_data.columns else 'N/A',
                        'Период': f"{city_data['date'].min().date()} - {city_data['date'].max().date()}"
                    }
                    city_stats_summary.append(stats)
                
                stats_df = pd.DataFrame(city_stats_summary)
                st.dataframe(stats_df, use_container_width=True)
        
        # Вкладки с визуализациями
        tab1, tab2, tab3 = st.tabs(["Быстрый анализ", "Scatter Plot", "Box & Violin Plots"])
        
        with tab1:
            st.subheader("Быстрый анализ признаков")
            
            if numeric_cols:
                selected_col = st.selectbox(
                    "Выберите признак для анализа:", 
                    numeric_cols[:15]
                )
                
                if selected_col in filtered_df.columns:
                    data = filtered_df[selected_col]
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Среднее", f"{data.mean():.2f}")
                    with col2:
                        st.metric("Медиана", f"{data.median():.2f}")
                    with col3:
                        st.metric("Стд. отклонение", f"{data.std():.2f}")
                      
                    with col4:
                        # ДОБАВИТЬ ТОЧНОСТЬ СРЕДНЕГО (95% доверительный интервал)
                        n = len(data)
                        if n > 1:
                            se = data.std() / np.sqrt(n)  # стандартная ошибка
                            ci = 1.96 * se  # 95% доверительный интервал
                            st.metric("Точность среднего (±)", f"±{ci:.3f}")
                        else:
                            st.metric("Диапазон", f"{data.min():.1f}-{data.max():.1f}")
                                        
                      # Гистограмма
                      fig = px.histogram(
                          filtered_df, 
                          x=selected_col, 
                          nbins=30,
                          title=f"Распределение {selected_col} - {selected_city}",
                          color='city_name' if selected_city == "Все города" and 'city_name' in filtered_df.columns else None
                          )
                      st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Scatter Plot Analysis")
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis:", numeric_cols, index=0)
                with col2:
                    y_col = st.selectbox("Y-axis:", numeric_cols, index=min(1, len(numeric_cols)-1))
                
                if x_col and y_col:
                    # Создаем копию данных для графика
                    plot_data = filtered_df.copy()
                    
                    # Конвертируем даты в числовой формат если нужно
                    if x_col == 'date' or (x_col in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df[x_col])):
                        plot_data[x_col] = convert_dates_to_numeric(plot_data[x_col])
                    
                    if y_col == 'date' or (y_col in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df[y_col])):
                        plot_data[y_col] = convert_dates_to_numeric(plot_data[y_col])
                    
                    # Простой scatter plot с полупрозрачными точками
                    fig = go.Figure()
                    
                    # Если выбраны все города, добавляем цвет по городам
                    if selected_city == "Все города" and 'city_name' in plot_data.columns:
                        cities = plot_data['city_name'].unique()
                        colors = px.colors.qualitative.Set1
                        
                        for i, city in enumerate(cities[:10]):  # Ограничиваем 10 городами для читаемости
                            city_data = plot_data[plot_data['city_name'] == city]
                            color = colors[i % len(colors)]
                            
                            fig.add_trace(go.Scatter(
                                x=city_data[x_col],
                                y=city_data[y_col],
                                mode='markers',
                                name=city,
                                marker=dict(
                                    color=color,
                                    size=6,
                                    opacity=0.6
                                )
                            ))
                    else:
                        # Для одного города простые точки
                        fig.add_trace(go.Scatter(
                            x=plot_data[x_col],
                            y=plot_data[y_col],
                            mode='markers',
                            name='Данные',
                            marker=dict(
                                color='rgba(100, 100, 100, 0.3)',  # Серый с прозрачностью 30%
                                size=6
                            ),
                            opacity=0.3
                        ))
                    
                    # Добавляем линию регрессии
                    if st.checkbox("Показать линию регрессии", value=True):
                        mask = ~np.isnan(plot_data[x_col]) & ~np.isnan(plot_data[y_col])
                        x_clean = plot_data[x_col][mask].values
                        y_clean = plot_data[y_col][mask].values
                        
                        if len(x_clean) > 1:
                            coeffs = np.polyfit(x_clean, y_clean, 1)
                            x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
                            y_pred = coeffs[0] * x_range + coeffs[1]
                            
                            # Яркая, четкая линия регрессии
                            fig.add_trace(go.Scatter(
                                x=x_range,
                                y=y_pred,
                                mode='lines',
                                name='Линия регрессии',
                                line=dict(
                                    color='red',
                                    width=3,
                                    dash='solid'
                                ),
                                opacity=1.0
                            ))
                    
                    fig.update_layout(
                        title=f"{y_col} vs {x_col} - {selected_city}",
                        xaxis_title=x_col,
                        yaxis_title=y_col
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Box Plot и Violin Plot")
            
            if numeric_cols:
                # Выбор признака для анализа распределения
                box_col = st.selectbox("Признак для анализа распределения:", numeric_cols[:10])
                
                if box_col in filtered_df.columns:
                    # Создаем копию данных для графика
                    plot_data = filtered_df.copy()
                    
                    # Конвертируем даты в числовой формат если нужно
                    if box_col == 'date' or (box_col in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df[box_col])):
                        plot_data[box_col] = convert_dates_to_numeric(plot_data[box_col])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Box Plot
                        if selected_city == "Все города" and 'city_name' in plot_data.columns:
                            fig_box = px.box(
                                plot_data,
                                y=box_col,
                                x='city_name',
                                title=f"Box Plot: {box_col} по городам"
                            )
                        else:
                            fig_box = px.box(
                                plot_data,
                                y=box_col,
                                title=f"Box Plot: {box_col} - {selected_city}"
                            )
                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    with col2:
                        # Violin Plot
                        if selected_city == "Все города" and 'city_name' in plot_data.columns:
                            fig_violin = px.violin(
                                plot_data,
                                y=box_col,
                                x='city_name',
                                title=f"Violin Plot: {box_col} по городам",
                                box=True
                            )
                        else:
                            fig_violin = px.violin(
                                plot_data,
                                y=box_col,
                                title=f"Violin Plot: {box_col} - {selected_city}",
                                box=True
                            )
                        st.plotly_chart(fig_violin, use_container_width=True)

# ==========================================================
# PAGE 2 — АНАЛИЗ ДАННЫХ
# ==========================================================
elif page == "Анализ данных":
    
    if filtered_df.empty:
        st.error("Для анализа нужны данные.")
    else:
        # Показываем информацию о выбранном городе
        if selected_city == "Все города":
            st.header(f"Анализ данных: Все города")
        else:
            st.header(f"Анализ данных: {selected_city}")
        
        numeric_cols = get_numeric_columns(filtered_df)
        
        if not numeric_cols:
            st.error("Нет числовых признаков для анализа.")
        else:
            st.write(f"**Доступно признаков:** {len(numeric_cols)}")
            
            analysis_method = st.selectbox(
                "Метод анализа:",
                ["Регрессия", "Кластеризация", "PCA"],
                index=0
            )
            
            if analysis_method in ["Регрессия", "Кластеризация", "PCA"]:
                df_scaled = prepare_scaled_data(filtered_df, numeric_cols)
            
            if analysis_method == "Регрессия":
                st.header("Регрессионный анализ")
                
                target = st.selectbox(
                    "Целевая переменная (Y):", 
                    numeric_cols[:10]
                )
                
                if target:
                    if len(numeric_cols) > 1:
                        correlations = filtered_df[numeric_cols].corr()[target].abs().sort_values(ascending=False)
                        correlations = correlations[correlations.index != target]
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
                    
                    # Конвертируем даты если нужно
                    if target in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df[target]):
                        # Для дат используем числовое представление
                        X = df_scaled[features]
                        y = convert_dates_to_numeric(filtered_df[target])
                        y_scaled = (y - y.mean()) / y.std()
                    else:
                        X = df_scaled[features]
                        y = df_scaled[target]
                    
                    sample_size = min(2000, len(X))
                    if len(X) > sample_size:
                        sample_idx = np.random.choice(len(X), sample_size, replace=False)
                        X_sample = X.iloc[sample_idx]
                        y_sample = y.iloc[sample_idx] if hasattr(y, 'iloc') else y[sample_idx]
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_sample, y_sample, test_size=test_size, random_state=42
                        )
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                    
                    models_config = {
                        "Линейная регрессия": LinearRegression(),
                        "Гребневая регрессия": Ridge(alpha=1.0),
                        "Лассо регрессия": Lasso(alpha=0.01),
                        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
                        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42)
                    }
                    
                    results = {}
                    
                    for name, model in models_config.items():
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        results[name] = {
                            'R²': r2_score(y_test, y_pred),
                            'MAE': mean_absolute_error(y_test, y_pred),
                            'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100,
                            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
                        }
                    
                    # Сравнительная таблица
                    st.subheader("Сравнение моделей")
                    results_df = pd.DataFrame(results).T.round(4)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Визуализация лучшей модели
                    best_model_name = max(results.keys(), key=lambda x: results[x]['R²'])
                    best_model = models_config[best_model_name]
                    best_model.fit(X_train, y_train)
                    
                    st.subheader(f"Лучшая модель: {best_model_name}")
                    
                    # Создаем график с четко видимыми линиями
                    fig = go.Figure()
                    
                    # Точки с низкой прозрачностью
                    fig.add_trace(go.Scatter(
                        x=y_test,
                        y=best_model.predict(X_test),
                        mode='markers',
                        name='Прогнозы',
                        marker=dict(
                            color='#A7FC00',
                            size=6
                        ),
                        opacity=0.3
                    ))
                    
                    # Линия идеального прогноза - яркая и четкая
                    min_val = min(y_test.min(), best_model.predict(X_test).min())
                    max_val = max(y_test.max(), best_model.predict(X_test).max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Идеально',
                        line=dict(
                            dash='dash',
                            color='red',
                            width=3
                        ),
                        opacity=1.0
                    ))
                    
                    fig.update_layout(
                        title=f"Фактические vs Предсказанные значения - {selected_city}",
                        xaxis_title="Фактические значения",
                        yaxis_title="Предсказанные значения"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_method == "Кластеризация":
                st.header("Кластеризация")
                
                if len(numeric_cols) >= 2:
                    default_features = numeric_cols[:2]
                else:
                    default_features = numeric_cols
                
                features = st.multiselect(
                    "Признаки для кластеризации:",
                    numeric_cols[:6],
                    default=default_features
                )
                
                if len(features) >= 2:
                    algorithm = st.selectbox("Алгоритм:", ["K-Means", "DBSCAN"])
                    
                    if algorithm == "K-Means":
                        n_clusters = st.slider("Кластеров:", 2, 6, 3)
                    else:
                        eps = st.slider("EPS:", 0.1, 1.0, 0.5, 0.1)
                    
                    X = df_scaled[features]
                    
                    sample_size = min(1000, len(X))
                    if len(X) > sample_size:
                        X_sample = X.sample(sample_size, random_state=42)
                    else:
                        X_sample = X
                    
                    if algorithm == "K-Means":
                        model = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
                        clusters = model.fit_predict(X_sample)
                        st.metric("Inertia", f"{model.inertia_:.2f}")
                    else:
                        model = DBSCAN(eps=eps, min_samples=5)
                        clusters = model.fit_predict(X_sample)
                    
                    df_viz = filtered_df.loc[X_sample.index].copy()
                    df_viz['Cluster'] = clusters
                    
                    # Конвертируем даты если нужно для первого признака
                    if features[0] in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df[features[0]]):
                        df_viz[features[0]] = convert_dates_to_numeric(df_viz[features[0]])
                    
                    fig = px.scatter(
                        df_viz,
                        x=features[0],
                        y=features[1],
                        color='Cluster',
                        title=f"Кластеризация: {features[0]} vs {features[1]} - {selected_city}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            else:  # PCA
                st.header("Анализ главных компонент (PCA)")
                
                if len(numeric_cols) >= 4:
                    pca_features = numeric_cols[:4]
                else:
                    pca_features = numeric_cols
                
                if len(pca_features) >= 2:
                    n_components = min(3, len(pca_features))
                    
                    X = df_scaled[pca_features]
                    
                    sample_size = min(1000, len(X))
                    if len(X) > sample_size:
                        X_sample = X.sample(sample_size, random_state=42)
                    else:
                        X_sample = X
                    
                    pca = PCA(n_components=n_components)
                    X_pca = pca.fit_transform(X_sample)
                    
                    explained_var = pca.explained_variance_ratio_
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[f"PC{i+1}" for i in range(n_components)],
                        y=explained_var,
                        name='Доля дисперсии'
                    ))
                    
                    fig.update_layout(
                        title=f"Объясненная дисперсия - {selected_city}",
                        yaxis_title='Доля дисперсии'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if n_components >= 2:
                        df_viz = filtered_df.loc[X_sample.index].copy()
                        df_viz['PC1'] = X_pca[:, 0]
                        df_viz['PC2'] = X_pca[:, 1]
                        
                        fig_scatter = px.scatter(
                            df_viz,
                            x='PC1',
                            y='PC2',
                            color='city_name' if selected_city == "Все города" and 'city_name' in df_viz.columns else None,
                            title=f"PCA - Проекция данных - {selected_city}"
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================================
# PAGE 3 — ПРОГНОЗИРОВАНИЕ
# ==========================================================
else:  # Прогнозирование
    
    if filtered_df.empty:
        st.error("Для прогнозирования нужны данные.")
    else:
        # Показываем информацию о выбранном городе
        if selected_city == "Все города":
            st.header(f"Прогнозирование временных рядов: Все города")
        else:
            st.header(f"Прогнозирование временных рядов: {selected_city}")
        
        # Проверяем наличие даты
        if 'date' not in filtered_df.columns:
            st.error("В данных отсутствует колонка с датами (date)")
        else:
            numeric_cols = get_numeric_columns(filtered_df)
            
            if not numeric_cols:
                st.error("Нет числовых признаков для прогнозирования.")
            else:
                # Конфигурация прогнозирования
                col1, col2 = st.columns(2)
                
                with col1:
                    target_col = st.selectbox(
                        "Целевая переменная для прогноза:",
                        numeric_cols[:10]
                    )
                
                with col2:
                    forecast_days = st.slider("Дней для прогноза:", 7, 90, 30)
                
                # Предупреждение если выбраны "Все города"
                if selected_city == "Все города":
                    st.warning("⚠️ Для прогнозирования выбран режим 'Все города'. Анализ будет проводиться по сводным данным всех городов.")
                
                # Подготовка данных
                if target_col:
                    ts_data = prepare_time_series_data(filtered_df, target_col)
                    
                    if ts_data is not None:
                        # Информация о временном ряде
                        st.subheader("Информация о временном ряде")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Дней данных", len(ts_data))
                        with col2:
                            # Конвертируем дату в строку перед отображением
                            start_date = ts_data['ds'].min()
                            if hasattr(start_date, 'date'):
                                start_date_str = str(start_date.date())
                            else:
                                start_date_str = str(start_date)
                            st.metric("Начало", start_date_str)
                        with col3:
                            # Конвертируем дату в строку перед отображением
                            end_date = ts_data['ds'].max()
                            if hasattr(end_date, 'date'):
                                end_date_str = str(end_date.date())
                            else:
                                end_date_str = str(end_date)
                            st.metric("Конец", end_date_str)
                        
                        # Визуализация исходных данных
                        fig_original = px.line(
                            ts_data,
                            x='ds',
                            y='y',
                            title=f"Исходный временной ряд: {target_col} - {selected_city}",
                            line_shape='linear'
                        )
                        
                        st.plotly_chart(fig_original, use_container_width=True)
                        
                        # Выбор метода прогнозирования
                        st.subheader("Методы прогнозирования")
                        
                        models_to_use = st.multiselect(
                            "Выберите модели для сравнения:",
                            ["ARIMA", "Exponential Smoothing"],
                            default=["ARIMA", "Exponential Smoothing"]
                        )
                        
                        if models_to_use:
                            forecasts = {}
                            models_info = {}
                            
                            # Прогнозирование выбранными методами
                            for model_name in models_to_use:
                                with st.spinner(f"Обучение {model_name}..."):
                                    if model_name == "ARIMA":
                                        model_fit, forecast = arima_forecast(
                                            ts_data, 
                                            periods=forecast_days,
                                            order=(1,1,1)
                                        )
                                    elif model_name == "Exponential Smoothing":
                                        model_fit, forecast = exponential_smoothing_forecast(
                                            ts_data,
                                            periods=forecast_days
                                        )
                                    else:
                                        continue
                                        
                                    if forecast is not None:
                                        forecasts[model_name] = forecast
                                        models_info[model_name] = model_fit
                            
                            # Визуализация прогнозов
                            if forecasts:
                                st.subheader("Сравнение прогнозов")
                                
                                fig_forecast = go.Figure()
                                
                                # Исходные данные
                                fig_forecast.add_trace(go.Scatter(
                                    x=ts_data['ds'],
                                    y=ts_data['y'],
                                    mode='lines',
                                    name='Исходные данные',
                                    line=dict(color='#1f77b4', width=2.5)
                                ))
                                
                                # Прогнозы
                                line_colors = [
                                    '#ff7f0e',  # Оранжевый (для ARIMA)
                                    '#2ca02c',  # Зеленый (для Exponential Smoothing)
                                ]
                                line_styles = ['solid', 'dash']
                                
                                for idx, (model_name, forecast_df) in enumerate(forecasts.items()):
                                    color = line_colors[idx % len(line_colors)]
                                    style = line_styles[idx % len(line_styles)]
                                    
                                    if model_name in ["ARIMA", "Exponential Smoothing"]:
                                        fig_forecast.add_trace(go.Scatter(
                                            x=forecast_df['ds'],
                                            y=forecast_df['yhat'],
                                            mode='lines',
                                            name=f'Прогноз {model_name}',
                                            line=dict(
                                                color=color,
                                                width=3,
                                                dash=style
                                            )
                                        ))
                                
                                fig_forecast.update_layout(
                                    title=f"Прогноз {target_col} на {forecast_days} дней - {selected_city}",
                                    xaxis_title="Дата",
                                    yaxis_title=target_col,
                                    plot_bgcolor='white',
                                    paper_bgcolor='white'
                                )
                                
                                st.plotly_chart(fig_forecast, use_container_width=True)
                                
                                # Таблица с последними прогнозами
                                st.subheader("Последние значения прогнозов")
                                
                                # Создаем общий DataFrame для всех прогнозов
                                forecast_table = pd.DataFrame()
                                
                                # Собираем все прогнозы в один DataFrame
                                for idx, (model_name, forecast_df) in enumerate(forecasts.items()):
                                    if model_name not in ["ARIMA", "Exponential Smoothing"]:
                                        continue
                                        
                                    temp_df = forecast_df[['ds', 'yhat']].copy()
                                    temp_df.columns = ['Дата', model_name]
                                    
                                    if forecast_table.empty:
                                        forecast_table = temp_df.set_index('Дата')
                                    else:
                                        temp_df = temp_df.set_index('Дата')
                                        forecast_table = forecast_table.join(temp_df, how='outer')
                                
                                # Сортируем по дате и показываем последние 10 значений
                                if not forecast_table.empty:
                                    forecast_table = forecast_table.sort_index(ascending=False)
                                    st.dataframe(
                                        forecast_table.head(10).round(2), 
                                        use_container_width=True
                                    )
                                    
                                    # Показываем статистику по прогнозам
                                    st.subheader("Статистика прогнозов")
                                    stats_df = pd.DataFrame()
                                    for model_name, forecast_df in forecasts.items():
                                        if model_name not in ["ARIMA", "Exponential Smoothing"]:
                                            continue
                                            
                                        stats_df[model_name] = [
                                            forecast_df['yhat'].mean(),
                                            forecast_df['yhat'].std(),
                                            forecast_df['yhat'].min(),
                                            forecast_df['yhat'].max()
                                        ]
                                    
                                    stats_df.index = ['Среднее', 'Стд. отклонение', 'Минимум', 'Максимум']
                                    st.dataframe(stats_df.round(2), use_container_width=True)

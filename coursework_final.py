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
                           mean_absolute_percentage_error, mean_squared_error,
                           accuracy_score, explained_variance_score)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score

# Для прогнозирования временных рядов
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(
    page_title="Weather Analytics Dashboard", 
    layout="wide"
)

# ----------------------------------------------------------
# НОВЫЕ ФУНКЦИИ ДЛЯ ОЦЕНКИ ТОЧНОСТИ
# ----------------------------------------------------------
def calculate_confidence_interval(data, confidence=0.95):
    """Вычисляет доверительный интервал для среднего"""
    n = len(data)
    if n <= 1:
        return 0
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # несмещенная оценка
    se = std / np.sqrt(n)  # стандартная ошибка
    
    # Z-score для доверительного уровня
    from scipy import stats
    t_score = stats.t.ppf((1 + confidence) / 2, df=n-1)
    
    return t_score * se

def calculate_forecast_backtesting(ts_data, model_type='ARIMA', horizons=[1, 7, 30]):
    """Backtesting точности прогнозов на разных горизонтах"""
    results = {}
    
    for horizon in horizons:
        if len(ts_data) < horizon * 2:
            continue
        
        mae_scores = []
        mape_scores = []
        
        # Скользящее окно для backtesting
        for i in range(len(ts_data) - horizon - 10):
            train = ts_data.iloc[i:i+10]
            test = ts_data.iloc[i+10:i+10+horizon]
            
            if len(test) < horizon:
                continue
            
            try:
                if model_type == "ARIMA":
                    model = ARIMA(train.set_index('ds')['y'], order=(1,1,1))
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=horizon)
                else:
                    model = ExponentialSmoothing(
                        train.set_index('ds')['y'],
                        seasonal_periods=min(7, len(train)),
                        trend='add',
                        seasonal='add'
                    )
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=horizon)
                
                mae = mean_absolute_error(test['y'], forecast)
                mape = mean_absolute_percentage_error(test['y'], forecast) * 100
                
                mae_scores.append(mae)
                mape_scores.append(mape)
                
            except:
                continue
        
        if mae_scores:
            results[horizon] = {
                'MAE_mean': np.mean(mae_scores),
                'MAE_std': np.std(mae_scores),
                'MAPE_mean': np.mean(mape_scores),
                'MAPE_std': np.std(mape_scores),
                'Accuracy_mean': 100 - np.mean(mape_scores),
                'n_tests': len(mae_scores)
            }
    
    return results

def calculate_model_confidence(y_true, y_pred):
    """Вычисляет различные метрики точности для модели"""
    metrics = {}
    
    # Основные метрики
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['MAPE'] = mean_absolute_percentage_error(y_true, y_pred) * 100
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['R2'] = r2_score(y_true, y_pred)
    
    # Accuracy для классификации (если порог)
    if len(np.unique(y_true)) < 10:  # Для дискретных значений
        # Округляем до ближайшего целого для классификации
        y_pred_rounded = np.round(y_pred)
        metrics['Accuracy_rounded'] = accuracy_score(y_true, y_pred_rounded)
    
    # Процент точности на основе MAPE
    metrics['Accuracy_percentage'] = max(0, 100 - metrics['MAPE'])
    
    # Доверительный интервал для точности
    n = len(y_true)
    if n > 1:
        residuals = y_true - y_pred
        se_residuals = np.std(residuals) / np.sqrt(n)
        metrics['CI_95'] = 1.96 * se_residuals
    
    return metrics

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
    
    return daily_df

# Загрузка данных
daily_df = load_data()

# ----------------------------------------------------------
# ФУНКЦИИ ДЛЯ ФИЛЬТРАЦИИ ПО ГОРОДАМ
# ----------------------------------------------------------
def get_available_cities(df):
    """Получить список доступных городов"""
    if df.empty or 'city_name' not in df.columns:
        return []
    cities = sorted(df['city_name'].dropna().unique().tolist())
    return cities

def filter_data_by_city(df, selected_city):
    """Фильтрация данных по выбранному городу"""
    if df.empty or not selected_city:
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
            index=0 if len(available_cities) > 0 else 0,
            help="Выберите город для анализа"
        )
        
        # Фильтруем данные по выбранному городу
        filtered_df = filter_data_by_city(daily_df, selected_city)
        
        # Показываем статистику по выбранному городу
        city_stats = get_city_stats(daily_df, selected_city)
        if city_stats:
            st.sidebar.info(f"""
            **Статистика {selected_city}:**
            - Дней данных: {city_stats['total_days']}
            - Период: {city_stats['date_range']}
            - Ср. температура: {city_stats['avg_temp']}°C
            - Ср. осадки: {city_stats['avg_precipitation']} мм
            """)
        
        numeric_cols = get_numeric_columns(filtered_df)
        st.sidebar.success(f"{selected_city}: {len(filtered_df):,} записей, {len(numeric_cols)} признаков")
    else:
        st.sidebar.warning("Колонка 'city_name' не найдена в данных")
        filtered_df = daily_df
        selected_city = "Все данные"
        numeric_cols = get_numeric_columns(filtered_df)
else:
    st.sidebar.error("Данные не загружены или нет колонки с городами")
    filtered_df = daily_df
    selected_city = "Все данные"
    numeric_cols = []

# ==========================================================
# PAGE 1 — ВИЗУАЛИЗАЦИЯ ДАННЫХ (С ACCURACY)
# ==========================================================
if page == "Визуализация данных":
    
    if filtered_df.empty:
        st.error("Данные не загружены.")
    else:
        st.header(f"Визуализация данных: {selected_city}")
        
        numeric_cols = get_numeric_columns(filtered_df)
        
        # Быстрые KPI
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'city_name' in filtered_df.columns:
                st.metric("Выбран город", selected_city)
            else:
                st.metric("Записей", len(filtered_df))
        
        with col2:
            if 'date' in filtered_df.columns:
                unique_days = len(filtered_df['date'].dt.date.unique())
                st.metric("Дней данных", unique_days)
        
        with col3:
            st.metric("Признаков", len(numeric_cols))
        
        # Вкладки с визуализациями
        tab1, tab2, tab3 = st.tabs(["Быстрый анализ с Accuracy", "Scatter Plot", "Box & Violin Plots"])
        
        with tab1:
            st.subheader("Быстрый анализ признаков с оценкой точности")
            
            if numeric_cols:
                selected_col = st.selectbox(
                    "Выберите признак для анализа:", 
                    numeric_cols[:15]
                )
                
                if selected_col in filtered_df.columns:
                    data = filtered_df[selected_col].dropna()
                    
                    if len(data) > 0:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            mean_val = data.mean()
                            ci = calculate_confidence_interval(data.values)
                            st.metric(
                                "Среднее ± CI95", 
                                f"{mean_val:.2f}",
                                delta=f"±{ci:.3f}"
                            )
                        
                        with col2:
                            median_val = data.median()
                            st.metric("Медиана", f"{median_val:.2f}")
                        
                        with col3:
                            std_val = data.std()
                            cv = (std_val / mean_val * 100) if mean_val != 0 else 0
                            st.metric(
                                "Стд. отклонение", 
                                f"{std_val:.2f}",
                                delta=f"CV: {cv:.1f}%"
                            )
                        
                        with col4:
                            range_val = f"{data.min():.1f}-{data.max():.1f}"
                            iqr = data.quantile(0.75) - data.quantile(0.25)
                            st.metric(
                                "Диапазон (IQR)", 
                                range_val,
                                delta=f"IQR: {iqr:.2f}"
                            )
                        
                        # Дополнительные метрики точности
                        st.subheader("Дополнительные метрики точности")
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            # Коэффициент вариации
                            cv_percent = (std_val / abs(mean_val)) * 100 if mean_val != 0 else 0
                            st.metric("Коэф. вариации", f"{cv_percent:.1f}%")
                        
                        with col_b:
                            # Стандартная ошибка
                            se = std_val / np.sqrt(len(data))
                            st.metric("Стд. ошибка", f"{se:.3f}")
                        
                        with col_c:
                            # Доля выбросов (по правилу 3σ)
                            outliers = data[(data < mean_val - 3*std_val) | (data > mean_val + 3*std_val)]
                            outlier_percent = len(outliers) / len(data) * 100
                            st.metric("Выбросы (>3σ)", f"{outlier_percent:.1f}%")
                        
                        # Гистограмма с доверительными интервалами
                        fig = px.histogram(
                            filtered_df, 
                            x=selected_col, 
                            nbins=30,
                            title=f"Распределение {selected_col} - {selected_city}",
                            marginal="box"
                        )
                        
                        # Добавляем линии для среднего и доверительного интервала
                        fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                                    annotation_text=f"Среднее: {mean_val:.2f}")
                        fig.add_vrect(x0=mean_val-ci, x1=mean_val+ci, 
                                    fillcolor="green", opacity=0.2, 
                                    annotation_text=f"95% ДИ: ±{ci:.3f}")
                        
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
                    
                    # Удаляем пропуски
                    plot_data = plot_data[[x_col, y_col]].dropna()
                    
                    if len(plot_data) > 1:
                        # Вычисляем корреляцию и ее значимость
                        correlation = plot_data[x_col].corr(plot_data[y_col])
                        
                        # Простой scatter plot
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=plot_data[x_col],
                            y=plot_data[y_col],
                            mode='markers',
                            name='Данные',
                            marker=dict(
                                color='rgba(100, 100, 100, 0.3)',
                                size=6
                            ),
                            opacity=0.3
                        ))
                        
                        # Добавляем линию регрессии
                        if st.checkbox("Показать линию регрессии", value=True):
                            x_clean = plot_data[x_col].values
                            y_clean = plot_data[y_col].values
                            
                            coeffs = np.polyfit(x_clean, y_clean, 1)
                            x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
                            y_pred = coeffs[0] * x_range + coeffs[1]
                            
                            # Яркая, четкая линия регрессии
                            fig.add_trace(go.Scatter(
                                x=x_range,
                                y=y_pred,
                                mode='lines',
                                name=f'Регрессия (r={correlation:.3f})',
                                line=dict(
                                    color='red',
                                    width=3,
                                    dash='solid'
                                ),
                                opacity=1.0
                            ))
                        
                        fig.update_layout(
                            title=f"{y_col} vs {x_col} - Корреляция: {correlation:.3f}",
                            xaxis_title=x_col,
                            yaxis_title=y_col
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Показываем статистику корреляции
                        with st.expander("Статистика корреляции"):
                            col_c1, col_c2, col_c3 = st.columns(3)
                            with col_c1:
                                st.metric("Коэф. корреляции", f"{correlation:.3f}")
                            with col_c2:
                                # R² для линейной регрессии
                                if len(x_clean) > 1:
                                    r_squared = correlation ** 2
                                    st.metric("R²", f"{r_squared:.3f}")
                            with col_c3:
                                # Количество наблюдений
                                st.metric("Наблюдений", len(plot_data))
        
        with tab3:
            st.subheader("Box Plot и Violin Plot")
            
            if numeric_cols:
                box_col = st.selectbox("Признак для анализа распределения:", numeric_cols[:10])
                
                if box_col in filtered_df.columns:
                    plot_data = filtered_df.copy()
                    
                    if box_col == 'date' or (box_col in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df[box_col])):
                        plot_data[box_col] = convert_dates_to_numeric(plot_data[box_col])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_box = px.box(
                            plot_data,
                            y=box_col,
                            title=f"Box Plot: {box_col} - {selected_city}"
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    with col2:
                        fig_violin = px.violin(
                            plot_data,
                            y=box_col,
                            title=f"Violin Plot: {box_col} - {selected_city}",
                            box=True
                        )
                        st.plotly_chart(fig_violin, use_container_width=True)

# ==========================================================
# PAGE 2 — АНАЛИЗ ДАННЫХ (С ACCURACY)
# ==========================================================
elif page == "Анализ данных":
    
    if filtered_df.empty:
        st.error("Для анализа нужны данные.")
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
                st.header("Регрессионный анализ с оценкой точности")
                
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
                        
                        # Вычисляем все метрики точности
                        metrics = calculate_model_confidence(y_test, y_pred)
                        results[name] = metrics
                    
                    # Сравнительная таблица
                    st.subheader("Сравнение моделей (метрики точности)")
                    results_df = pd.DataFrame(results).T.round(4)
                    
                    # Переименовываем колонки для лучшей читаемости
                    results_df.columns = ['MAE', 'MAPE (%)', 'RMSE', 'R²', 'Accuracy (%)', 'CI 95%']
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Визуализация лучшей модели по Accuracy
                    best_model_name = max(results.keys(), key=lambda x: results[x]['Accuracy_percentage'])
                    best_model = models_config[best_model_name]
                    best_model.fit(X_train, y_train)
                    
                    st.subheader(f"Лучшая модель по точности: {best_model_name}")
                    st.info(f"Точность: {results[best_model_name]['Accuracy_percentage']:.2f}%")
                    
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
                    
                    # Линия идеального прогноза
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
                    
                    # Дополнительный анализ остатков
                    with st.expander("Анализ остатков модели"):
                        residuals = y_test - best_model.predict(X_test)
                        
                        col_r1, col_r2 = st.columns(2)
                        
                        with col_r1:
                            fig_res = px.histogram(
                                x=residuals,
                                title="Распределение остатков",
                                nbins=30
                            )
                            fig_res.add_vline(x=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_res, use_container_width=True)
                        
                        with col_r2:
                            fig_scatter = px.scatter(
                                x=best_model.predict(X_test),
                                y=residuals,
                                title="Остатки vs Прогнозы",
                                labels={'x': 'Прогнозы', 'y': 'Остатки'}
                            )
                            fig_scatter.add_hline(y=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_sc

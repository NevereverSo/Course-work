import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (silhouette_score, r2_score, mean_absolute_error, 
                           mean_absolute_percentage_error, mean_squared_error,
                           accuracy_score, classification_report)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(
    page_title="Weather Analytics Dashboard", 
    layout="wide"
)

@st.cache_data(ttl=3600, show_spinner="Загрузка данных...")
def load_data():
    """Быстрая загрузка данных"""
    try:
        daily_df = pd.read_csv("daily_weather_smallest.csv", low_memory=False)
        if not daily_df.empty and 'date' in daily_df.columns:
            daily_df['date'] = pd.to_datetime(daily_df['date'], errors='coerce')
    except Exception as e:
        st.warning(f"Ошибка загрузки daily_weather_smallest.csv: {str(e)}")
        daily_df = pd.DataFrame()
    
    try:
        cities_df = pd.read_csv("cities.csv", low_memory=False)
    except Exception as e:
        st.warning(f"Ошибка загрузки cities.csv: {str(e)}")
        cities_df = pd.DataFrame()
    
    try:
        countries_df = pd.read_csv("countries.csv", low_memory=False)
    except Exception as e:
        st.warning(f"Ошибка загрузки countries.csv: {str(e)}")
        countries_df = pd.DataFrame()
    
    return daily_df, cities_df, countries_df

@st.cache_data(ttl=1800, max_entries=5)
def prepare_time_series_data(df, target_col, date_col='date'):
    """
    Подготовка данных временного ряда с оптимизацией
    """
    if df.empty or target_col not in df.columns or date_col not in df.columns:
        return None
    
    ts_df = df.sort_values(date_col).drop_duplicates(subset=[date_col])
    
    if len(ts_df) < 10:
        return None
    
    ts_data = ts_df[[date_col, target_col]].copy()
    ts_data.columns = ['ds', 'y']
    
    ts_data['y'] = ts_data['y'].interpolate(method='linear')
    
    zero_threshold_vars = ['precipitation', 'snow', 'depth', 'rain', 'snow_depth']
    if any(var in target_col.lower() for var in zero_threshold_vars):
        ts_data['y'] = ts_data['y'] + np.random.uniform(0.01, 0.1, len(ts_data))
    
    return ts_data

@st.cache_data(ttl=1800, max_entries=3)
def arima_forecast(ts_data, periods=30, order=(1,1,1)):
    """
    Быстрое прогнозирование ARIMA
    """
    try:
        ts_series = ts_data.set_index('ds')['y']
        
        max_points = 50
        if len(ts_series) > max_points:
            ts_series = ts_series.iloc[-max_points:]
        
        model = ARIMA(ts_series, order=order)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model_fit = model.fit(method_kwargs={'maxiter': 30})
            except:
                st.info("ARIMA (1,1,1) не работает, пробуем (1,0,0)")
                model = ARIMA(ts_series, order=(1,0,0))
                model_fit = model.fit(method_kwargs={'maxiter': 20})
        
        forecast = model_fit.forecast(steps=periods)
        last_date = ts_series.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        forecast_df = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecast.values
        })
        
        return model_fit, forecast_df
    except Exception as e:
        st.error(f"ARIMA не сработала: {str(e)[:100]}")
        return None, None

@st.cache_data(ttl=1800, max_entries=3)
def exponential_smoothing_forecast(ts_data, periods=30):
    """
    ПРОСТАЯ И РАБОЧАЯ версия Exponential Smoothing
    """
    try:
        ts_series = ts_data.set_index('ds')['y']
        
        if len(ts_series) < 5:
            st.warning("Слишком мало данных для Exponential Smoothing")
            return None, None
        
        max_points = 60
        if len(ts_series) > max_points:
            ts_series = ts_series.iloc[-max_points:]
        
        if len(ts_series) >= 14:
            seasonal_periods = 7
        else:
            seasonal_periods = None
        
        try:
            if seasonal_periods and len(ts_series) >= 2 * seasonal_periods:
                model = ExponentialSmoothing(
                    ts_series,
                    seasonal_periods=seasonal_periods,
                    trend='add',
                    seasonal='add',
                    initialization_method='estimated'
                )
            else:
                model = ExponentialSmoothing(
                    ts_series,
                    seasonal=None,
                    trend='add',
                    initialization_method='estimated'
                )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model_fit = model.fit(optimized=True)
                except:
                    # Если оптимизация не работает, используем ручные параметры
                    model_fit = model.fit(
                        smoothing_level=0.3,
                        smoothing_trend=0.1,
                        smoothing_seasonal=0.1 if seasonal_periods else None,
                        optimized=False
                    )
            
            forecast = model_fit.forecast(steps=periods)
            
            if np.any(np.isnan(forecast)):
                st.warning("Прогноз содержит NaN, используем простой прогноз")
                raise ValueError("NaN in forecast")
            
        except Exception as e:
            st.info("Используем простую модель ETS")
            model = ExponentialSmoothing(
                ts_series,
                seasonal=None,
                trend=None,
                initialization_method='estimated'
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
        st.error(f"Exponential Smoothing не сработала: {str(e)[:100]}")
        return None, None

def simple_forecast_fallback(ts_data, periods=30):
    """
    Простой прогноз для случаев, когда сложные модели не работают
    """
    try:
        ts_series = ts_data.set_index('ds')['y']
        
        if len(ts_series) < 2:
            forecast_values = np.zeros(periods)
        else:
            last_value = ts_series.iloc[-1]
            
            if len(ts_series) >= 5:
                recent = ts_series.iloc[-5:].values
                if len(recent) >= 2:
                    x = np.arange(len(recent))
                    coeffs = np.polyfit(x, recent, 1)
                    trend = coeffs[0]
                    
                    forecast_values = last_value + trend * np.arange(1, periods + 1)
                else:
                    forecast_values = np.full(periods, last_value)
            else:
                forecast_values = np.full(periods, last_value)
        
        last_date = ts_data['ds'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        return pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecast_values
        })
    except:
        return pd.DataFrame({
            'ds': pd.date_range(start='2023-01-01', periods=periods, freq='D'),
            'yhat': np.zeros(periods)
        })

def evaluate_time_series_model(ts_data, model_type='arima', test_size=0.3):
    if ts_data is None or len(ts_data) < 30:
        return None, None
    
    split_idx = int(len(ts_data) * (1 - test_size))
    train_data = ts_data.iloc[:split_idx].copy()
    test_data = ts_data.iloc[split_idx:].copy()
    
    if model_type == 'arima':
        _, forecast = arima_forecast(train_data, periods=len(test_data))
    elif model_type == 'exponential':
        _, forecast = exponential_smoothing_forecast(train_data, periods=len(test_data))
    else:
        return None, None
    
    if forecast is None:
        return None, None
    
    y_true = test_data['y'].values
    y_pred = forecast['yhat'].values[:len(y_true)]
    
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    if len(y_true) < 3:
        return None, None
    
    metrics = {}
    
    if len(y_true) > 1:
        naive_forecast = np.zeros_like(y_true)
        naive_forecast[1:] = y_true[:-1]
        naive_forecast[0] = y_true[0]
        
        mae_naive = np.mean(np.abs(y_true[1:] - naive_forecast[1:]))
        mae_model = np.mean(np.abs(y_true - y_pred))
        
        if mae_naive > 0:
            metrics['MASE'] = mae_model / mae_naive
            metrics['Улучшение (%)'] = ((mae_naive - mae_model) / mae_naive) * 100
        else:
            metrics['MASE'] = np.nan
            metrics['Улучшение (%)'] = np.nan
    else:
        metrics['MASE'] = np.nan
        metrics['Улучшение (%)'] = np.nan
    
    if 'MASE' in metrics and not np.isnan(metrics['MASE']):
        if metrics['MASE'] < 1:
            metrics['R²'] = 1 - metrics['MASE']
        else:
            metrics['R²'] = -(metrics['MASE'] - 1)
    else:
        metrics['R²'] = np.nan
    
    metrics['MAE'] = np.mean(np.abs(y_true - y_pred))
    metrics['RMSE'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    mask = y_true != 0
    if np.sum(mask) > 0:
        metrics['MAPE (%)'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        metrics['MAPE (%)'] = np.nan
    
    return metrics, (y_true, y_pred, test_data['ds'].values)
  
def calculate_metrics(y_true, y_pred, variable_name=""):
    import numpy as np
    
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) < 3:
        return {'MASE': np.nan, 'Улучшение (%)': np.nan, 'MAE': np.nan, 
                'RMSE': np.nan, 'sMAPE (%)': np.nan, 'R²': np.nan}
    
    metrics = {}
    
    if len(y_true) > 1:
        naive = np.zeros_like(y_true)
        naive[1:] = y_true[:-1]
        naive[0] = y_true[0]
        
        mae_model = np.mean(np.abs(y_true - y_pred))
        mae_naive = np.mean(np.abs(y_true[1:] - naive[1:]))
        
        if mae_naive > 0:
            mase = mae_model / mae_naive
            metrics['MASE'] = float(mase)
            metrics['Улучшение (%)'] = float((1 - mase) * 100)
        else:
            metrics['MASE'] = np.nan
            metrics['Улучшение (%)'] = np.nan
    else:
        metrics['MASE'] = np.nan
        metrics['Улучшение (%)'] = np.nan
    
    metrics['MAE'] = float(np.mean(np.abs(y_true - y_pred)))
    metrics['RMSE'] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator[denominator == 0] = 0.001
    smape = 100 * np.mean(np.abs(y_pred - y_true) / denominator)
    metrics['sMAPE (%)'] = float(min(smape, 200))
    
    if 'MASE' in metrics and metrics['MASE'] is not np.nan:
        if metrics['MASE'] < 1:
            metrics['R²'] = float(1 - metrics['MASE'])
        else:
            metrics['R²'] = float(-(metrics['MASE'] - 1))
    else:
        metrics['R²'] = np.nan
    
    return metrics

daily_df = pd.DataFrame()
cities_df = pd.DataFrame()
countries_df = pd.DataFrame()

try:
    daily_df, cities_df, countries_df = load_data()
except Exception as e:
    st.error(f"Ошибка при загрузке данных: {str(e)}")
    daily_df = pd.DataFrame()
    cities_df = pd.DataFrame()
    countries_df = pd.DataFrame()

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

@st.cache_data
def get_numeric_columns(df):
    """Быстрое получение числовых колонок"""
    if df.empty:
        return []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
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
    
    sample_size = min(3000, len(_df))
    if len(_df) > sample_size:
        df_sample = _df.sample(sample_size, random_state=42)
        scaler.fit(df_sample[numeric_cols])
    else:
        scaler.fit(_df[numeric_cols])
    
    df_scaled = _df.copy()
    df_scaled[numeric_cols] = scaler.transform(_df[numeric_cols])
    
    return df_scaled

@st.cache_data
def prepare_classification_data(df, target_col, features):
    """Подготовка данных для классификации"""
    if df.empty or target_col not in df.columns:
        return None, None, None, None
    
    median_val = df[target_col].median()
    y = (df[target_col] > median_val).astype(int)
    
    X = df[features].copy()
    
    X = X.fillna(X.mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, median_val
  
def convert_dates_to_numeric(dates):
    """Конвертирует даты в числовой формат (количество дней с первой даты)"""
    if len(dates) == 0:
        return dates
    
    if np.issubdtype(dates.dtype, np.number):
        return dates
    
    if pd.api.types.is_datetime64_any_dtype(dates):
        min_date = dates.min()
        numeric_dates = (dates - min_date).dt.days
        return numeric_dates
    elif hasattr(dates.iloc[0], 'date'):
        min_date = min(dates)
        numeric_dates = [(date - min_date).days for date in dates]
        return pd.Series(numeric_dates)
    
    return dates
  
st.sidebar.title("Weather Analytics")

page = st.sidebar.radio(
    "Навигация",
    ["Визуализация данных", "Анализ данных", "Прогнозирование"]
)

st.sidebar.subheader("Выбор города")

if 'daily_df' in locals() and not daily_df.empty and 'city_name' in daily_df.columns:
    available_cities = get_available_cities(daily_df)
    
    if available_cities:
        selected_city = st.sidebar.selectbox(
            "Выберите город для анализа:",
            options=available_cities,
            index=0,
            help="Выберите город для анализа. 'Все города' показывает сводную статистику."
        )
        
        filtered_df = filter_data_by_city(daily_df, selected_city)
        
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
    filtered_df = daily_df if 'daily_df' in locals() else pd.DataFrame()
    selected_city = "Все города"
    numeric_cols = []

# ... (предыдущий код без изменений до страницы "Визуализация данных")

if page == "Визуализация данных":
    
    if filtered_df.empty:
        st.error("Данные не загружены. Проверьте наличие файла daily_weather_smallest.csv")
    else:
        if selected_city == "Все города":
            st.header(f"Визуализация данных: Все города")
        else:
            st.header(f"Визуализация данных: {selected_city}")
        
        numeric_cols = get_numeric_columns(filtered_df)
        
        col1, col2, col3, col4 = st.columns(4)
        
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
            
        with col4:
            if len(numeric_cols) > 0:
                data_accuracy = filtered_df[numeric_cols].notna().mean().mean() * 100
                st.metric("Точность данных", f"{data_accuracy:.1f}%")
            else:
                st.metric("Точность данных", "N/A")
        
        # НОВАЯ СЕКЦИЯ: Методы info() и describe()
        with st.expander("📊 Методы info() и describe() для анализа данных", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Метод info()")
                st.write("**Общая информация о датасете:**")
                
                # Создаем строку с информацией о датасете
                info_text = f"""
                **Размер данных:** {filtered_df.shape[0]} строк × {filtered_df.shape[1]} столбцов
                
                **Типы данных:**
                """
                
                # Получаем информацию о типах данных
                dtypes_info = filtered_df.dtypes.value_counts()
                for dtype, count in dtypes_info.items():
                    info_text += f"\n- {dtype}: {count} колонок"
                
                # Информация о пропусках
                missing_info = filtered_df.isnull().sum()
                total_missing = missing_info.sum()
                info_text += f"\n\n**Пропуски:** {total_missing:,} пропущенных значений"
                
                st.info(info_text)
                
                # Показываем типы данных по колонкам
                if st.checkbox("Показать типы данных для всех колонок"):
                    dtype_df = pd.DataFrame({
                        'Колонка': filtered_df.columns,
                        'Тип данных': filtered_df.dtypes.values,
                        'Не-null значения': filtered_df.notna().sum().values,
                        'Пропуски': filtered_df.isnull().sum().values
                    })
                    st.dataframe(dtype_df, use_container_width=True)
            
            with col2:
                st.subheader("Метод describe()")
                
                # Выбираем числовые колонки для describe
                if len(numeric_cols) > 0:
                    # Ограничиваем количество колонок для лучшего отображения
                    display_cols = numeric_cols[:8]  # Первые 8 числовых колонок
                    
                    describe_df = filtered_df[display_cols].describe().round(2)
                    
                    # Переименовываем индексы на русский
                    describe_df.index = ['Количество', 'Среднее', 'Стд. отклонение', 
                                       'Минимум', '25%', '50% (медиана)', '75%', 'Максимум']
                    
                    st.dataframe(describe_df, use_container_width=True)
                    
                    # Дополнительная статистика
                    if st.checkbox("Показать дополнительную статистику"):
                        st.write("**Корреляция между признаками:**")
                        # Вычисляем корреляцию только для выбранных колонок
                        if len(display_cols) > 1:
                            corr_matrix = filtered_df[display_cols].corr().round(3)
                            # Визуализация тепловой карты корреляции
                            fig = px.imshow(
                                corr_matrix,
                                text_auto=True,
                                aspect="auto",
                                title="Матрица корреляции",
                                color_continuous_scale='RdBu_r',
                                range_color=[-1, 1]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Нет числовых колонок для метода describe()")
        
        # Кнопка для экспорта статистики
        if st.button("📥 Экспорт статистики в CSV"):
            if len(numeric_cols) > 0:
                # Создаем DataFrame со статистикой
                stats_list = []
                for col in numeric_cols[:15]:  # Ограничиваем количество колонок
                    if col in filtered_df.columns:
                        stats = filtered_df[col].describe()
                        stats_list.append({
                            'Признак': col,
                            'Количество': stats['count'],
                            'Среднее': round(stats['mean'], 2),
                            'Стд. отклонение': round(stats['std'], 2),
                            'Минимум': round(stats['min'], 2),
                            '25%': round(stats['25%'], 2),
                            'Медиана': round(stats['50%'], 2),
                            '75%': round(stats['75%'], 2),
                            'Максимум': round(stats['max'], 2),
                            'Пропуски': filtered_df[col].isnull().sum(),
                            'Точность (%)': round((filtered_df[col].notna().sum() / len(filtered_df)) * 100, 1)
                        })
                
                stats_df = pd.DataFrame(stats_list)
                
                # Создаем CSV
                csv = stats_df.to_csv(index=False, encoding='utf-8-sig')
                
                # Кнопка для скачивания
                st.download_button(
                    label="Скачать статистику CSV",
                    data=csv,
                    file_name=f"weather_stats_{selected_city}.csv",
                    mime="text/csv",
                )
        
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
                        'Точность данных (%)': round(city_data[numeric_cols].notna().mean().mean() * 100, 1) if len(numeric_cols) > 0 else 'N/A',
                        'Период': f"{city_data['date'].min().date()} - {city_data['date'].max().date()}"
                    }
                    city_stats_summary.append(stats)
                
                stats_df = pd.DataFrame(city_stats_summary)
                st.dataframe(stats_df, use_container_width=True)
        
        # ИЗМЕНЯЕМ ВКЛАДКИ - добавляем новую вкладку для info/describe
        tab1, tab2, tab3, tab4 = st.tabs(["Быстрый анализ", "Scatter Plot", "Box & Violin Plots", "Распределения"])
        
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
                        accuracy_pct = data.notna().sum() / len(data) * 100
                        st.metric("Точность данных", f"{accuracy_pct:.1f}%")
                    
                    # Дополнительная статистика для выбранной колонки
                    with st.expander("📈 Подробная статистика для выбранного признака"):
                        col_stats = filtered_df[selected_col].describe()
                        stats_df = pd.DataFrame({
                            'Метрика': ['Количество', 'Среднее', 'Стд. отклонение', 'Минимум', '25%', '50% (медиана)', '75%', 'Максимум'],
                            'Значение': [col_stats['count'], 
                                       round(col_stats['mean'], 4),
                                       round(col_stats['std'], 4),
                                       round(col_stats['min'], 4),
                                       round(col_stats['25%'], 4),
                                       round(col_stats['50%'], 4),
                                       round(col_stats['75%'], 4),
                                       round(col_stats['max'], 4)]
                        })
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # Показываем информацию о пропусках
                        missing_count = filtered_df[selected_col].isnull().sum()
                        if missing_count > 0:
                            st.warning(f"⚠️ В колонке '{selected_col}' {missing_count} пропущенных значений ({round(missing_count/len(filtered_df)*100, 1)}%)")
                    
                    fig = px.histogram(
                            filtered_df, 
                            x=selected_col, 
                            nbins=30,
                            title=f"Распределение {selected_col} - {selected_city}",
                            color='city_name' if selected_city == "Все города" and 'city_name' in filtered_df.columns else None
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # ... (существующий код для Scatter Plot без изменений)
        
        with tab3:
            # ... (существующий код для Box & Violin Plots без изменений)
            
        # НОВАЯ ВКЛАДКА ДЛЯ РАСПРЕДЕЛЕНИЙ
        with tab4:
            st.subheader("Анализ распределений признаков")
            
            if numeric_cols:
                # Выбор нескольких признаков для сравнения
                selected_features = st.multiselect(
                    "Выберите признаки для сравнения распределений:",
                    numeric_cols[:10],
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                )
                
                if selected_features:
                    # Создаем графики распределений
                    fig = go.Figure()
                    
                    colors = px.colors.qualitative.Set1
                    
                    for i, feature in enumerate(selected_features):
                        # Нормализуем данные для лучшего сравнения
                        data = filtered_df[feature].dropna()
                        if len(data) > 0:
                            # Гистограмма
                            fig.add_trace(go.Histogram(
                                x=data,
                                name=feature,
                                opacity=0.6,
                                marker_color=colors[i % len(colors)],
                                nbinsx=30
                            ))
                    
                    fig.update_layout(
                        title=f"Сравнение распределений признаков - {selected_city}",
                        xaxis_title="Значения",
                        yaxis_title="Частота",
                        barmode='overlay',
                        bargap=0.1
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Сводная таблица статистики для выбранных признаков
                    st.subheader("Сводная статистика")
                    summary_stats = []
                    for feature in selected_features:
                        if feature in filtered_df.columns:
                            data = filtered_df[feature]
                            stats = data.describe()
                            summary_stats.append({
                                'Признак': feature,
                                'Среднее': round(stats['mean'], 3),
                                'Медиана': round(stats['50%'], 3),
                                'Стд. отклонение': round(stats['std'], 3),
                                'Минимум': round(stats['min'], 3),
                                'Максимум': round(stats['max'], 3),
                                'Пропуски': data.isnull().sum(),
                                'Уникальных значений': data.nunique()
                            })
                    
                    if summary_stats:
                        summary_df = pd.DataFrame(summary_stats)
                        st.dataframe(summary_df, use_container_width=True)

elif page == "Анализ данных":
    
    if filtered_df.empty:
        st.error("Для анализа нужны данные.")
    else:
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
                ["Регрессия", "Классификация", "Кластеризация", "PCA"],
                index=0
            )
            
            if analysis_method in ["Регрессия", "Классификация", "Кластеризация", "PCA"]:
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
                    test_size = st.slider("Тестовая выборка:", 0.1, 0.8, 0.2, 0.05)
                    
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
                        "Лассо регрессия": Lasso(alpha=0.01)
                    }
                    
                    results = {}
                    
                    for name, model in models_config.items():
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        metrics = calculate_metrics(y_test, y_pred, target)
                        
                        results[name] = {
                            'R²': metrics['R²'],
                            'MAE': metrics['MAE'],
                            'RMSE': metrics['RMSE'],
                            'MAPE (%)': metrics.get('MAPE (%)', 'N/A'),
                            'sMAPE (%)': metrics.get('sMAPE (%)', 'N/A')
                        }
                    
                    st.subheader("Сравнение моделей")
                    results_df = pd.DataFrame(results).T.round(4)
                    st.dataframe(results_df, use_container_width=True)
                    
                    best_model_name = max(results.keys(), key=lambda x: results[x]['R²'])
                    best_model = models_config[best_model_name]
                    best_model.fit(X_train, y_train)
                    
                    st.subheader(f"Лучшая модель: {best_model_name}")
                    
                    fig = go.Figure()
                  
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
            
            elif analysis_method == "Классификация":
                st.header("Классификация")
                
                target = st.selectbox(
                    "Целевая переменная для классификации:",
                    numeric_cols[:10]
                )
                
                if target:
                    median_val = filtered_df[target].median()
                    st.write(f"**Медиана {target}:** {median_val:.2f}")
                    st.write(f"**Классы:** 0 = ниже медианы, 1 = выше медианы")
                    
                    if len(numeric_cols) > 1:
                        correlations = filtered_df[numeric_cols].corr()[target].abs().sort_values(ascending=False)
                        correlations = correlations[correlations.index != target]
                        top_features = correlations.head(3).index.tolist()
                    else:
                        top_features = []
                    
                    features = st.multiselect(
                        "Признаки для классификации:",
                        numeric_cols,
                        default=top_features,
                        key="class_features"
                    )
                
                if target and features and len(features) > 0:
                    test_size = st.slider("Размер тестовой выборки:", 0.1, 0.4, 0.3, 0.05, key="class_test_size")
                    
                    X, y, scaler, median_val = prepare_classification_data(filtered_df, target, features)
                    
                    if X is not None:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42, stratify=y
                        )
                      
                        models_config = {
                            "Логистическая регрессия": LogisticRegression(random_state=42),
                            "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
                            "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
                            "SVM": SVC(kernel='rbf', probability=True, random_state=42)
                        }
                        
                        results = {}
                        
                        for name, model in models_config.items():
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                            
                            report = classification_report(y_test, y_pred, output_dict=True)
                            results[name] = {
                                'Accuracy': accuracy_score(y_test, y_pred),
                                'Precision': report['weighted avg']['precision'],
                                'Recall': report['weighted avg']['recall'],
                                'F1-Score': report['weighted avg']['f1-score']
                            }
                        
                        st.subheader("Сравнение моделей классификации")
                        results_df = pd.DataFrame(results).T.round(4)
                        st.dataframe(results_df, use_container_width=True)
                        
                        best_model_name = max(results.keys(), key=lambda x: results[x]['Accuracy'])
                        best_model = models_config[best_model_name]
                        best_model.fit(X_train, y_train)
                        
                        st.subheader(f"Лучшая модель: {best_model_name}")
                        st.metric("Accuracy", f"{results[best_model_name]['Accuracy']:.4f}")
                        
                        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
                        import matplotlib.pyplot as plt
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        cm = confusion_matrix(y_test, best_model.predict(X_test))
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ниже медианы', 'Выше медианы'])
                        disp.plot(cmap='Blues', ax=ax)
                        ax.set_title(f"Матрица ошибок - {best_model_name}")
                        st.pyplot(fig)
                        
                        if hasattr(best_model, 'feature_importances_'):
                            st.subheader("Важность признаков")
                            importances = pd.DataFrame({
                                'Признак': features,
                                'Важность': best_model.feature_importances_
                            }).sort_values('Важность', ascending=False)
                            
                            fig = px.bar(
                                importances,
                                x='Важность',
                                y='Признак',
                                orientation='h',
                                title='Важность признаков для классификации'
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
                        
                        # Silhouette score для оценки качества кластеризации
                        if n_clusters > 1:
                            silhouette_avg = silhouette_score(X_sample, clusters)
                            st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
                    else:
                        model = DBSCAN(eps=eps, min_samples=5)
                        clusters = model.fit_predict(X_sample)
                    
                    if len(X_sample) > 0:
                        df_viz = pd.DataFrame(X_sample, columns=features)
                        df_viz['Cluster'] = clusters
                        
                        try:
                            valid_indices = X_sample.index[X_sample.index.isin(filtered_df.index)]
                            if len(valid_indices) > 0:
                                for col in ['city_name', 'date']:
                                    if col in filtered_df.columns:
                                        df_viz[col] = filtered_df.loc[valid_indices, col].values
                        except:
                            pass
                    else:
                        df_viz = pd.DataFrame()
                    
                    if not df_viz.empty:
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
                    
                    total_explained_var = sum(explained_var) * 100
                    st.metric("Объясненная дисперсия", f"{total_explained_var:.1f}%")
                    
                    if n_components >= 2:
                        df_viz = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
                        
                        try:
                            valid_indices = X_sample.index[X_sample.index.isin(filtered_df.index)]
                            if len(valid_indices) > 0:
                                for col in ['city_name', 'date']:
                                    if col in filtered_df.columns:
                                        df_viz[col] = filtered_df.loc[valid_indices, col].values
                        except:
                            pass
                        
                        fig_scatter = px.scatter(
                            df_viz,
                            x='PC1',
                            y='PC2',
                            color='city_name' if selected_city == "Все города" and 'city_name' in df_viz.columns else None,
                            title=f"PCA - Проекция данных - {selected_city}"
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

else:
    
    if filtered_df.empty:
        st.error("Для прогнозирования нужны данные.")
    else:
        if selected_city == "Все города":
            st.header(f"Прогнозирование временных рядов: Все города")
        else:
            st.header(f"Прогнозирование временных рядов: {selected_city}")
        
        if 'date' not in filtered_df.columns:
            st.error("В данных отсутствует колонка с датами (date)")
        else:
            numeric_cols = get_numeric_columns(filtered_df)
            
            if not numeric_cols:
                st.error("Нет числовых признаков для прогнозирования.")
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    target_col = st.selectbox(
                        "Целевая переменная для прогноза:",
                        numeric_cols[:10]
                    )
                
                with col2:
                    forecast_days = st.slider("Дней для прогноза:", 7, 90, 30)
                
                with col3:
                    if target_col:
                        data_accuracy = filtered_df[target_col].notna().sum() / len(filtered_df) * 100
                        st.metric("Точность данных", f"{data_accuracy:.1f}%")
                
                if selected_city == "Все города":
                    st.warning("Режим 'Все города' - данные агрегированы")
                
                if target_col and target_col in filtered_df.columns:
                    with st.expander("Статистика выбранной переменной"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Среднее", f"{filtered_df[target_col].mean():.2f}")
                        with col2:
                            st.metric("Медиана", f"{filtered_df[target_col].median():.2f}")
                        with col3:
                            st.metric("Std", f"{filtered_df[target_col].std():.2f}")
                
                if target_col:
                    with st.spinner("Подготовка временного ряда..."):
                        ts_data = prepare_time_series_data(filtered_df, target_col)
                    
                    if ts_data is not None:
                        st.subheader("Информация о временном ряде")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Дней данных", len(ts_data))
                        with col2:
                            start_date = ts_data['ds'].min()
                            st.metric("Начало", str(start_date.date()))
                        with col3:
                            end_date = ts_data['ds'].max()
                            st.metric("Конец", str(end_date.date()))
                        with col4:
                            ts_accuracy = ts_data['y'].notna().sum() / len(ts_data) * 100
                            st.metric("Точность ряда", f"{ts_accuracy:.1f}%")
                        
                        fig_original = px.line(
                            ts_data,
                            x='ds',
                            y='y',
                            title=f"Исходный временной ряд: {target_col} - {selected_city}",
                            line_shape='linear'
                        )
                        st.plotly_chart(fig_original, use_container_width=True)
                        
                        st.subheader("Методы прогнозирования")
                        
                        fast_mode = st.checkbox("Быстрый режим (ограниченные данные)", value=True)
                        
                        models_to_use = st.multiselect(
                            "Выберите модели для сравнения:",
                            ["ARIMA", "Exponential Smoothing"],
                            default=["ARIMA", "Exponential Smoothing"]
                        )
                        
                        if models_to_use:
                            forecasts = {}
                            evaluation_results = {}
                            
                            for model_name in models_to_use:
                                with st.spinner(f"Оценка точности {model_name}..."):
                                    metrics, test_data = evaluate_time_series_model(
                                        ts_data, 
                                        model_type='arima' if model_name == 'ARIMA' else 'exponential',
                                        test_size=0.3
                                    )
                                    
                                    if metrics:
                                        evaluation_results[model_name] = metrics
                                        
                                        if test_data:
                                            y_true, y_pred, dates = test_data
                                            
                                            fig_test = go.Figure()
                                            fig_test.add_trace(go.Scatter(
                                                x=dates,
                                                y=y_true,
                                                mode='lines',
                                                name='Фактические значения',
                                                line=dict(color='blue', width=2)
                                            ))
                                            fig_test.add_trace(go.Scatter(
                                                x=dates,
                                                y=y_pred,
                                                mode='lines',
                                                name=f'Прогноз {model_name}',
                                                line=dict(color='red', width=2, dash='dash')
                                            ))
                                            
                                            fig_test.update_layout(
                                                title=f"Тестирование {model_name} (30% последних данных)",
                                                xaxis_title="Дата",
                                                yaxis_title=target_col
                                            )
                                            
                                            with st.expander(f"Оценка {model_name}"):
                                                st.plotly_chart(fig_test, use_container_width=True)
                                                
                                                metrics_df = pd.DataFrame([metrics]).T
                                                metrics_df.columns = ['Значение']
                                                st.dataframe(metrics_df.round(4))
                                
                                with st.spinner(f"Прогноз {model_name} на будущее..."):
                                    if model_name == "ARIMA":
                                        _, forecast = arima_forecast(ts_data, periods=forecast_days)
                                    else:
                                        _, forecast = exponential_smoothing_forecast(ts_data, periods=forecast_days)
                                    
                                    forecasts[model_name] = forecast
                            
                            if evaluation_results:
                                st.subheader("Результаты оценки точности")
                                
                                eval_df = pd.DataFrame(evaluation_results).T
                                
                                for col in eval_df.columns:
                                    if 'MAPE' in col or 'Improvement' in col:
                                        eval_df[col] = eval_df[col].apply(lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A")
                                    elif col == 'R²':
                                        eval_df[col] = eval_df[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
                                    else:
                                        eval_df[col] = eval_df[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
                                
                                st.dataframe(eval_df, use_container_width=True)
                            
                            if forecasts:
                                st.subheader("Сравнение прогнозов")
                                
                                fig_forecast = go.Figure()
                                
                                fig_forecast.add_trace(go.Scatter(
                                    x=ts_data['ds'],
                                    y=ts_data['y'],
                                    mode='lines',
                                    name='Исторические данные',
                                    line=dict(color='#1f77b4', width=2)
                                ))
                                
                                colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                                styles = ['solid', 'dash', 'dot', 'dashdot']
                                
                                for idx, (model_name, forecast_df) in enumerate(forecasts.items()):
                                    color = colors[idx % len(colors)]
                                    style = styles[idx % len(styles)]
                                    
                                    fig_forecast.add_trace(go.Scatter(
                                        x=forecast_df['ds'],
                                        y=forecast_df['yhat'],
                                        mode='lines',
                                        name=f'Прогноз {model_name}',
                                        line=dict(color=color, width=2.5, dash=style)
                                    ))
                                
                                fig_forecast.update_layout(
                                    title=f"Прогноз {target_col} на {forecast_days} дней - {selected_city}",
                                    xaxis_title="Дата",
                                    yaxis_title=target_col,
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig_forecast, use_container_width=True)
                                
                                st.subheader("Будущие значения")
                                
                                forecast_table = pd.DataFrame()
                                for model_name, forecast_df in forecasts.items():
                                    temp_df = forecast_df.copy()
                                    temp_df.columns = ['Дата', model_name]
                                    temp_df = temp_df.set_index('Дата')
                                    
                                    if forecast_table.empty:
                                        forecast_table = temp_df
                                    else:
                                        forecast_table = forecast_table.join(temp_df, how='outer')
                                
                                if not forecast_table.empty:
                                    forecast_table = forecast_table.sort_index()
                                    st.dataframe(
                                        forecast_table.round(2),
                                        use_container_width=True
                                    )
                                    
                                    st.subheader("Сводная статистика прогнозов")
                                    stats_data = []
                                    for model_name in forecasts.keys():
                                        values = forecast_table[model_name].dropna().values
                                        if len(values) > 0:
                                            stats_data.append([
                                                np.mean(values),
                                                np.std(values),
                                                np.min(values),
                                                np.max(values),
                                                (np.std(values) / max(abs(np.mean(values)), 0.001)) * 100
                                            ])
                                        else:
                                            stats_data.append(["N/A"] * 5)
                                    
                                    stats_df = pd.DataFrame(
                                        stats_data,
                                        index=forecasts.keys(),
                                        columns=['Среднее', 'Стд. отклонение', 'Минимум', 'Максимум', 'Коэф. вариации (%)']
                                    )
                                    st.dataframe(stats_df.round(2), use_container_width=True)

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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (silhouette_score, r2_score, mean_absolute_error, 
                           mean_absolute_percentage_error, mean_squared_error,
                           accuracy_score, classification_report)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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

# ----------------------------------------------------------
# ФУНКЦИИ ДЛЯ ПРОГНОЗИРОВАНИЯ ВРЕМЕННЫХ РЯДОВ (ИСПРАВЛЕННЫЕ)
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
    
    # Для переменных с нулевыми значениями добавляем шум чтобы избежать деления на 0
    zero_threshold_vars = ['precipitation', 'snow', 'depth', 'rain', 'snow_depth']
    if any(var in target_col.lower() for var in zero_threshold_vars):
        # Добавляем маленький шум вместо фиксированного значения
        ts_data['y'] = ts_data['y'] + np.random.uniform(0.01, 0.1, len(ts_data))
    
    return ts_data

@st.cache_data(ttl=1800, max_entries=3)
def arima_forecast(ts_data, periods=30, order=(1,1,1)):
    """
    Быстрое прогнозирование ARIMA
    """
    try:
        ts_series = ts_data.set_index('ds')['y']
        
        # СИЛЬНОЕ уменьшение данных для скорости
        max_points = 50  # Всего 50 точек для скорости
        if len(ts_series) > max_points:
            ts_series = ts_series.iloc[-max_points:]
            st.info(f"ARIMA: используем {max_points} последних точек")
        
        # Упрощенная ARIMA с фиксированными параметрами
        model = ARIMA(ts_series, order=order)
        
        # Быстрая подгонка с ограниченными итерациями
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model_fit = model.fit(method_kwargs={'maxiter': 30})
            except:
                # Если не получается, пробуем более простую модель
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
        
        # Уменьшаем данные
        max_points = 60  # 60 точек максимум
        if len(ts_series) > max_points:
            ts_series = ts_series.iloc[-max_points:]
        
        # ПРОСТЕЙШАЯ РАБОЧАЯ МОДЕЛЬ
        # 1. Определяем сезонность по длине данных
        if len(ts_series) >= 14:  # Хотя бы 2 недели
            seasonal_periods = 7  # Недельная сезонность
        else:
            seasonal_periods = None
        
        # 2. Пробуем простую модель
        try:
            if seasonal_periods and len(ts_series) >= 2 * seasonal_periods:
                # Модель с сезонностью
                model = ExponentialSmoothing(
                    ts_series,
                    seasonal_periods=seasonal_periods,
                    trend='add',
                    seasonal='add',
                    initialization_method='estimated'
                )
            else:
                # Модель без сезонности
                model = ExponentialSmoothing(
                    ts_series,
                    seasonal=None,
                    trend='add',
                    initialization_method='estimated'
                )
            
            # Подгоняем модель с упрощенными параметрами
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Пробуем подогнать с начальными параметрами
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
            
            # Прогноз
            forecast = model_fit.forecast(steps=periods)
            
            # Проверяем прогноз
            if np.any(np.isnan(forecast)):
                st.warning("Прогноз содержит NaN, используем простой прогноз")
                raise ValueError("NaN in forecast")
            
        except Exception as e:
            # РЕЗЕРВНЫЙ ВАРИАНТ: простейшая модель
            st.info("Используем простую модель ETS")
            model = ExponentialSmoothing(
                ts_series,
                seasonal=None,
                trend=None,
                initialization_method='estimated'
            )
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=periods)
        
        # Создаем DataFrame
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
            # Нет данных
            forecast_values = np.zeros(periods)
        else:
            # Наивный прогноз: последнее значение + тренд
            last_value = ts_series.iloc[-1]
            
            # Рассчитываем простой тренд
            if len(ts_series) >= 5:
                recent = ts_series.iloc[-5:].values
                if len(recent) >= 2:
                    # Линейный тренд
                    x = np.arange(len(recent))
                    coeffs = np.polyfit(x, recent, 1)
                    trend = coeffs[0]  # Ежедневное изменение
                    
                    # Прогноз с трендом
                    forecast_values = last_value + trend * np.arange(1, periods + 1)
                else:
                    forecast_values = np.full(periods, last_value)
            else:
                forecast_values = np.full(periods, last_value)
        
        # Создаем DataFrame
        last_date = ts_data['ds'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        return pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecast_values
        })
    except:
        # Аварийный вариант
        return pd.DataFrame({
            'ds': pd.date_range(start='2023-01-01', periods=periods, freq='D'),
            'yhat': np.zeros(periods)
        })
# ----------------------------------------------------------
# ИСПРАВЛЕННЫЕ МЕТРИКИ ДЛЯ ВРЕМЕННЫХ РЯДОВ
# ----------------------------------------------------------
def calculate_time_series_metrics(y_true, y_pred, variable_name=""):
    """
    Правильный расчет метрик для временных рядов
    """
    metrics = {}
    
    # Преобразуем в массивы
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Фильтруем NaN
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) < 2:
        return {
            'RMSE': np.nan,
            'MAE': np.nan,
            'R²': np.nan,
            'MAPE (%)': np.nan,
            'sMAPE (%)': np.nan
        }
    
    # 1. Базовые метрики
    try:
        metrics['RMSE'] = float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)))
        metrics['MAE'] = float(mean_absolute_error(y_true_clean, y_pred_clean))
    except:
        metrics['RMSE'] = np.nan
        metrics['MAE'] = np.nan
    
    # 2. R² ДЛЯ ВРЕМЕННЫХ РЯДОВ (правильный расчет)
    try:
        # Для временных рядов сравниваем с наивным прогнозом (последнее значение)
        naive_forecast = np.roll(y_true_clean, 1)
        naive_forecast[0] = y_true_clean[0]  # Первое значение такое же
        
        # SS_res для нашей модели
        ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
        # SS_res для наивной модели
        ss_res_naive = np.sum((y_true_clean - naive_forecast) ** 2)
        
        if ss_res_naive == 0:
            metrics['R²'] = np.nan
        else:
            # R² относительно наивной модели
            r2 = 1 - (ss_res / ss_res_naive)
            # Ограничиваем в разумных пределах
            metrics['R²'] = float(max(min(r2, 1.0), -1.0))
    except:
        metrics['R²'] = np.nan
    
    # 3. Процентные ошибки
    zero_sensitive_vars = ['precipitation', 'snow', 'depth', 'rain', 'snow_depth', 'solar']
    
    if any(var in variable_name.lower() for var in zero_sensitive_vars):
        # Для переменных с нулями используем sMAPE
        try:
            smape = safe_smape(y_true_clean, y_pred_clean)
            metrics['sMAPE (%)'] = float(smape) if not np.isnan(smape) else np.nan
            metrics['MAPE (%)'] = "N/A"
        except:
            metrics['sMAPE (%)'] = np.nan
            metrics['MAPE (%)'] = "N/A"
    else:
        # Для остальных - MAPE
        try:
            mape = safe_mape(y_true_clean, y_pred_clean)
            if not np.isnan(mape):
                metrics['MAPE (%)'] = float(mape)
                metrics['sMAPE (%)'] = np.nan
            else:
                metrics['MAPE (%)'] = "N/A"
                metrics['sMAPE (%)'] = float(safe_smape(y_true_clean, y_pred_clean))
        except:
            metrics['MAPE (%)'] = np.nan
            metrics['sMAPE (%)'] = np.nan
    
    return metrics
# ----------------------------------------------------------
# НОВЫЕ ФУНКЦИИ ДЛЯ ОЦЕНКИ ТОЧНОСТИ ПРОГНОЗИРОВАНИЯ
# ----------------------------------------------------------
def safe_mape(y_true, y_pred):
    """
    Безопасный расчет MAPE с обработкой нулевых и близких к нулю значений
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Фильтруем нулевые значения
    mask = (y_true != 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
    
    if np.sum(mask) == 0:
        return np.nan
    
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # Рассчитываем MAPE только для ненулевых значений
    ape = np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered) * 100
    
    # Отсекаем выбросы (более 500%)
    ape = ape[ape <= 500]
    
    if len(ape) == 0:
        return np.nan
    
    return np.mean(ape)

def safe_smape(y_true, y_pred):
    """
    Расчет sMAPE (Symmetric MAPE) - более устойчивая метрика
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return np.nan
    
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Избегаем деления на ноль
    denominator[denominator == 0] = np.finfo(float).eps
    
    smape_values = 100 * numerator / denominator
    
    # Отсекаем выбросы
    smape_values = smape_values[smape_values <= 200]
    
    if len(smape_values) == 0:
        return np.nan
    
    return np.mean(smape_values)

def safe_r2_score(y_true, y_pred):
    """
    Безопасный расчет R² с обработкой крайних случаев
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) < 2:
        return np.nan
    
    # Проверяем, что данные не все одинаковые
    if np.all(y_true == y_true[0]):
        return np.nan
    
    # Рассчитываем R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Избегаем деления на ноль
    if ss_tot == 0:
        return np.nan
    
    r2 = 1 - (ss_res / ss_tot)
    
    # Ограничиваем R² в разумных пределах
    return max(min(r2, 1.0), -1.0)

def calculate_forecast_metrics(y_true, y_pred, variable_name=""):
    """
    Расчет всех метрик точности прогнозирования с учетом особенностей переменной
    """
    metrics = {}
    
    # Проверка данных
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Фильтруем NaN
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) < 2:
        return {
            'RMSE': np.nan,
            'MAE': np.nan,
            'R²': np.nan,
            'MAPE (%)': np.nan,
            'sMAPE (%)': np.nan
        }
    
    # Базовые метрики с защитой от ошибок
    try:
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    except:
        metrics['RMSE'] = np.nan
    
    try:
        metrics['MAE'] = mean_absolute_error(y_true_clean, y_pred_clean)
    except:
        metrics['MAE'] = np.nan
    
    # R² score с защитой
    try:
        metrics['R²'] = safe_r2_score(y_true_clean, y_pred_clean)
    except:
        metrics['R²'] = np.nan
    
    # Для переменных с возможными нулевыми значениями используем sMAPE
    zero_sensitive_vars = ['precipitation', 'snow', 'depth', 'rain', 'snow_depth', 'solar']
    
    if any(var in variable_name.lower() for var in zero_sensitive_vars):
        try:
            metrics['sMAPE (%)'] = safe_smape(y_true_clean, y_pred_clean)
            metrics['MAPE (%)'] = "N/A (используйте sMAPE)"
        except:
            metrics['sMAPE (%)'] = np.nan
            metrics['MAPE (%)'] = "N/A"
    else:
        # Для остальных переменных можно использовать MAPE
        try:
            mape_val = safe_mape(y_true_clean, y_pred_clean)
            if not np.isnan(mape_val):
                metrics['MAPE (%)'] = mape_val
            else:
                metrics['MAPE (%)'] = "N/A (нулевые значения)"
                metrics['sMAPE (%)'] = safe_smape(y_true_clean, y_pred_clean)
        except:
            metrics['MAPE (%)'] = np.nan
            metrics['sMAPE (%)'] = np.nan
    
    return metrics

# ----------------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ ПЕРЕМЕННЫХ ПЕРЕД ОСНОВНЫМ КОДОМ
# ----------------------------------------------------------
# Инициализируем пустые DataFrame
daily_df = pd.DataFrame()
cities_df = pd.DataFrame()
countries_df = pd.DataFrame()

# Загрузка данных с обработкой ошибок
try:
    daily_df, cities_df, countries_df = load_data()
except Exception as e:
    st.error(f"Ошибка при загрузке данных: {str(e)}")
    daily_df = pd.DataFrame()
    cities_df = pd.DataFrame()
    countries_df = pd.DataFrame()

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
# ДОБАВЛЕНА ФУНКЦИЯ ДЛЯ КЛАССИФИКАЦИИ
# ----------------------------------------------------------
@st.cache_data
def prepare_classification_data(df, target_col, features):
    """Подготовка данных для классификации"""
    if df.empty or target_col not in df.columns:
        return None, None, None, None
    
    # Создаем бинарную целевую переменную (например, выше/ниже медианы)
    median_val = df[target_col].median()
    y = (df[target_col] > median_val).astype(int)
    
    # Отбираем признаки
    X = df[features].copy()
    
    # Заполняем пропуски
    X = X.fillna(X.mean())
    
    # Масштабирование
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, median_val

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

# ПРОВЕРКА: убедимся, что daily_df существует и не пустой
if 'daily_df' in locals() and not daily_df.empty and 'city_name' in daily_df.columns:
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
    filtered_df = daily_df if 'daily_df' in locals() else pd.DataFrame()
    selected_city = "Все города"
    numeric_cols = []

# ==========================================================
# PAGE 1 — ВИЗУАЛИЗАЦИЯ ДАННЫХ
# ==========================================================
if page == "Визуализация данных":
    
    if filtered_df.empty:
        st.error("Данные не загружены. Проверьте наличие файла daily_weather_smallest.csv")
    else:
        # Показываем информацию о выбранном городе
        if selected_city == "Все города":
            st.header(f"Визуализация данных: Все города")
        else:
            st.header(f"Визуализация данных: {selected_city}")
        
        numeric_cols = get_numeric_columns(filtered_df)
        
        # Быстрые KPI с accuracy
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
            # Точность данных (процент не пропущенных значений)
            if len(numeric_cols) > 0:
                data_accuracy = filtered_df[numeric_cols].notna().mean().mean() * 100
                st.metric("Точность данных", f"{data_accuracy:.1f}%")
            else:
                st.metric("Точность данных", "N/A")
        
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
                        'Точность данных (%)': round(city_data[numeric_cols].notna().mean().mean() * 100, 1) if len(numeric_cols) > 0 else 'N/A',
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
                        # Точность данных для этого признака (не пропущенных)
                        accuracy_pct = data.notna().sum() / len(data) * 100
                        st.metric("Точность данных", f"{accuracy_pct:.1f}%")
                                        
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
                                color='#FFC618',  # Серый с прозрачностью 30%
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
            
            # ДОБАВЛЕН ВЫБОР ТИПА АНАЛИЗА
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
                        "Лассо регрессия": Lasso(alpha=0.01)
                    }
                    
                    results = {}
                    
                    for name, model in models_config.items():
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        # Используем безопасный расчет метрик
                        metrics = calculate_forecast_metrics(y_test, y_pred, target)
                        
                        results[name] = {
                            'R²': metrics['R²'],
                            'MAE': metrics['MAE'],
                            'RMSE': metrics['RMSE'],
                            'MAPE (%)': metrics.get('MAPE (%)', 'N/A'),
                            'sMAPE (%)': metrics.get('sMAPE (%)', 'N/A')
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
            
            elif analysis_method == "Классификация":
                st.header("Классификация")
                
                # Выбор целевой переменной для классификации
                target = st.selectbox(
                    "Целевая переменная для классификации:",
                    numeric_cols[:10]
                )
                
                if target:
                    # Создаем бинарную классификацию
                    median_val = filtered_df[target].median()
                    st.write(f"**Медиана {target}:** {median_val:.2f}")
                    st.write(f"**Классы:** 0 = ниже медианы, 1 = выше медианы")
                    
                    # Выбор признаков
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
                    
                    # Подготовка данных для классификации
                    X, y, scaler, median_val = prepare_classification_data(filtered_df, target, features)
                    
                    if X is not None:
                        # Разделение данных
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42, stratify=y
                        )
                        
                        # Модели классификации
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
                            
                            # Расчет метрик
                            report = classification_report(y_test, y_pred, output_dict=True)
                            results[name] = {
                                'Accuracy': accuracy_score(y_test, y_pred),
                                'Precision': report['weighted avg']['precision'],
                                'Recall': report['weighted avg']['recall'],
                                'F1-Score': report['weighted avg']['f1-score']
                            }
                        
                        # Сравнительная таблица
                        st.subheader("Сравнение моделей классификации")
                        results_df = pd.DataFrame(results).T.round(4)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Визуализация лучшей модели
                        best_model_name = max(results.keys(), key=lambda x: results[x]['Accuracy'])
                        best_model = models_config[best_model_name]
                        best_model.fit(X_train, y_train)
                        
                        st.subheader(f"Лучшая модель: {best_model_name}")
                        st.metric("Accuracy", f"{results[best_model_name]['Accuracy']:.4f}")
                        
                        # Матрица ошибок
                        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
                        import matplotlib.pyplot as plt
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        cm = confusion_matrix(y_test, best_model.predict(X_test))
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ниже медианы', 'Выше медианы'])
                        disp.plot(cmap='Blues', ax=ax)
                        ax.set_title(f"Матрица ошибок - {best_model_name}")
                        st.pyplot(fig)
                        
                        # Важность признаков для tree-based моделей
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
                    
                    # ИСПРАВЛЕНИЕ ОШИБКИ: Используем правильный способ получения df_viz
                    if len(X_sample) > 0:
                        # Создаем DataFrame для визуализации
                        df_viz = pd.DataFrame(X_sample, columns=features)
                        df_viz['Cluster'] = clusters
                        
                        # Добавляем исходные данные если нужно
                        # Используем индекс из X_sample для безопасного доступа к filtered_df
                        try:
                            # Проверяем, что индексы существуют в filtered_df
                            valid_indices = X_sample.index[X_sample.index.isin(filtered_df.index)]
                            if len(valid_indices) > 0:
                                # Добавляем дополнительные данные если они есть
                                for col in ['city_name', 'date']:
                                    if col in filtered_df.columns:
                                        df_viz[col] = filtered_df.loc[valid_indices, col].values
                        except:
                            pass
                    else:
                        df_viz = pd.DataFrame()
                    
                    if not df_viz.empty:
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
                    
                    # ДОБАВЛЕНО: Общая объясненная дисперсия
                    total_explained_var = sum(explained_var) * 100
                    st.metric("Объясненная дисперсия", f"{total_explained_var:.1f}%")
                    
                    if n_components >= 2:
                        # Создаем DataFrame для визуализации
                        df_viz = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
                        
                        # Добавляем исходные данные если нужно
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

# ==========================================================
# PAGE 3 — ПРОГНОЗИРОВАНИЕ (ИСПРАВЛЕННАЯ ВЕРСИЯ)
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
                
                # Предупреждение
                if selected_city == "Все города":
                    st.warning("Режим 'Все города' - данные агрегированы")
                
                # Информация о данных
                if target_col and target_col in filtered_df.columns:
                    with st.expander("📊 Статистика выбранной переменной"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Среднее", f"{filtered_df[target_col].mean():.2f}")
                        with col2:
                            st.metric("Медиана", f"{filtered_df[target_col].median():.2f}")
                        with col3:
                            st.metric("Std", f"{filtered_df[target_col].std():.2f}")
                
                # Подготовка данных
                if target_col:
                    with st.spinner("Подготовка временного ряда..."):
                        ts_data = prepare_time_series_data(filtered_df, target_col)
                    
                    if ts_data is not None:
                        # Информация о ряде
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
                        
                        # Визуализация
                        fig_original = px.line(
                            ts_data,
                            x='ds',
                            y='y',
                            title=f"Исходный временной ряд: {target_col} - {selected_city}",
                            line_shape='linear'
                        )
                        st.plotly_chart(fig_original, use_container_width=True)
                        
                        # Выбор моделей
                        st.subheader("Методы прогнозирования")
                        
                        # Кнопка для быстрого/полного режима
                        fast_mode = st.checkbox("Быстрый режим (ограниченные данные)", value=True)
                        
                        models_to_use = st.multiselect(
                            "Выберите модели для сравнения:",
                            ["ARIMA", "Exponential Smoothing"],
                            default=["ARIMA", "Exponential Smoothing"]
                        )
                        
                        if models_to_use:
                            forecasts = {}
                            backtest_results = {}
                            
                            # Для каждой модели
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
                                    
                                    # Если модель не сработала, используем простой прогноз
                                    if forecast is None:
                                        st.warning(f"{model_name} не сработала, используем простой прогноз")
                                        forecast = simple_forecast_fallback(ts_data, forecast_days)
                                    
                                    forecasts[model_name] = forecast
                                    
                                    # Бэктестинг (если достаточно данных)
                                    if len(ts_data) > 30 and not fast_mode:
                                        try:
                                            # Разделяем данные
                                            split_idx = int(len(ts_data) * 0.7)
                                            train_data = ts_data.iloc[:split_idx]
                                            test_data = ts_data.iloc[split_idx:]
                                            
                                            # Обучаем на тренировочных данных
                                            if model_name == "ARIMA":
                                                _, test_forecast = arima_forecast(
                                                    train_data,
                                                    periods=len(test_data),
                                                    order=(1,1,1)
                                                )
                                            else:
                                                _, test_forecast = exponential_smoothing_forecast(
                                                    train_data,
                                                    periods=len(test_data)
                                                )
                                            
                                            if test_forecast is not None:
                                                # Сравниваем с тестовыми данными
                                                metrics = calculate_time_series_metrics(
                                                    test_data['y'].values,
                                                    test_forecast['yhat'].values,
                                                    target_col
                                                )
                                                backtest_results[model_name] = metrics
                                        except Exception as e:
                                            st.info(f"Бэктестинг для {model_name} пропущен: {str(e)[:50]}")
                            
                            # Визуализация прогнозов
                            if forecasts:
                                st.subheader("Сравнение прогнозов")
                                
                                # График
                                fig_forecast = go.Figure()
                                
                                # Исходные данные
                                fig_forecast.add_trace(go.Scatter(
                                    x=ts_data['ds'],
                                    y=ts_data['y'],
                                    mode='lines',
                                    name='Исторические данные',
                                    line=dict(color='#1f77b4', width=2)
                                ))
                                
                                # Цвета для прогнозов
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
                                
                                # Метрики точности (если есть бэктестинг)
                                if backtest_results:
                                    st.subheader("Точность прогнозирования (тестирование на 30% данных)")
                                    
                                    # Создаем таблицу
                                    backtest_df = pd.DataFrame(backtest_results).T
                                    
                                    # Форматируем
                                    def format_value(val):
                                        if isinstance(val, (int, float)):
                                            if np.isnan(val):
                                                return "N/A"
                                            elif abs(val) > 1e6:
                                                return "Ошибка"
                                            else:
                                                return f"{val:.4f}"
                                        else:
                                            return str(val)
                                    
                                    for col in backtest_df.columns:
                                        backtest_df[col] = backtest_df[col].apply(format_value)
                                    
                                    st.dataframe(backtest_df, use_container_width=True)
                                  
                                else:
                                    st.info("Для оценки точности отключите 'Быстрый режим'")
                                
                                # Таблица прогнозов
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
                                    
                                    # Сводная статистика
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_time_series(data:pd.DataFrame, category_code:str, city_name:str) -> pd.Series:
    return data.loc[(data['category_code'] == category_code) & (data['city'] == city_name)]['measurement_value']

def get_timestamp_index(data:pd.DataFrame)-> pd.DataFrame:
    data['base_time'] = data['base_time'].astype(str).apply(lambda x: x.zfill(4))
    data['timestamp'] = pd.to_datetime(data['base_date'] + ' ' + data['base_time'], format='%Y-%m-%d %H%M')
    data.set_index('timestamp', inplace=True)
    return data


def plot_weather_data(data:pd.Series, category:str, city:str) -> None:
    plt.figure(figsize=(15, 5))  # Set figure size
    plt.plot(data, label=f'{category}')  # 'bo-' = blue circles with lines

    # Customize the plot
    plt.xlabel('Time')
    plt.ylabel(f'{category}')
    plt.title(f'{category} on {city} Coupang Fulfillment Center')
    plt.grid(True, linestyle='--', alpha=0.6)  # Add grid lines
    plt.legend()

    # Show the plot
    plt.show()
    return 


def generate_delay_hours(row):
    delay = 0
    if row['PTY'] >= 1: # 강수타입이 0 (없음)이 아닌 모든 경우, 30분 딜레이
        delay += 0.5
    if row['RN1'] >= 1: # 강수량이 1mm 이상일 경우 15분 딜레이
        delay += 0.25
    if row['WSD'] >= 7: # 풍속이 7 m/s 이상의 강풍을 동반할 경우
        delay += 0.25
    if row['T1H'] < 0 or row['T1H'] > 30: # 온도가 영하 혹은 30도 이상 고온의 extreme 일 경우, 신선식품 배송에 영향
        delay += 0.5
    # 최대 지연시간 3시간으로 제한
    delay = min(delay, 3.0)
    # random noise 추가
    delay += np.random.uniform(-0.2, 0.2)
    return max(0, delay) # 음수 방지
    

def cleaning_data(data:pd.DataFrame, city_name:str) -> pd.DataFrame:
    data_pivot = data[data['city']==city_name].copy()
    data_common = data[data['city']==city_name].copy()

    data_common = data_common.groupby('timestamp').first()
    data_common = data_common[[ 'base_date', 'year', 'month', 'day', 'day_of_week','is_holiday', 'base_time',
                               'hour', 'nx', 'ny', 'admin_district_code','city', 'sub_address']]

    data_pivot['timestamp'] = data_pivot.index
    data_pivot = data_pivot.pivot(index='timestamp', columns='category_code', values='measurement_value')

    data_pivot['delay_hours'] = data_pivot.apply(generate_delay_hours, axis=1)
    data_merged = pd.merge(data_pivot, data_common, left_index=True, right_index=True, how='left')

    data_resampled = data_merged.resample('1h').first()

    # missing data cleansing 
    numeric_time_features = ['REH', 'RN1', 'T1H', 'UUU', 'VEC', 'VVV', 'WSD', 'delay_hours', 'base_time', 'hour']
    categorical_date_features = ['nx', 'ny', 'admin_district_code', 'city','sub_address','base_date', 'year', 'month', 'day', 'day_of_week', 'is_holiday']

    data_resampled[numeric_time_features] = data_resampled[numeric_time_features].interpolate(method='time')
    data_resampled[categorical_date_features] = data_resampled[categorical_date_features].interpolate(method='bfill')

    data_resampled['PTY'] = data_resampled['PTY'].astype('category')
    data_resampled['PTY'].fillna(data_resampled['PTY'].mode()[0], inplace=True)

    data_resampled.drop(['base_time','city','sub_address','UUU', 'VEC', 'VVV'], axis=1, inplace=True)
    return data_resampled



def feature_engineering(data:pd.DataFrame) -> pd.DataFrame:
    data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['is_weekend'] = data['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)  # 주말 여부
    data['day_of_week_encoded'] = pd.Categorical(data['day_of_week']).codes
    data['is_holiday'] = data['is_holiday'].astype(int)
    
    data['PTY_lag1'] = data['PTY'].shift(1)
    data['PTY_lag2'] = data['PTY'].shift(2)

    data['delay_hours_lag1'] = data['delay_hours'].shift(1)
    data['delay_hours_lag2'] = data['delay_hours'].shift(2)
    
    data = data.fillna(method='bfill')
    return data


def plot_prediction(y_preds:dict, y_true) -> None:
    fig, axes = plt.subplots(3,1, figsize=(14,12))
    y_test_sorted = pd.Series(y_true).sort_index()
    y_pred_LR = pd.Series(y_preds['y_pred_LinearRegression'],index=y_true.index).sort_index()
    y_pred_XGB = pd.Series(y_preds['y_pred_XGBoostRegressor'],index=y_true.index).sort_index()
    y_pred_RF = pd.Series(y_preds['y_pred_RandomForest'],index=y_true.index).sort_index()

    axes[0].plot(y_test_sorted, '-', label='Raw Data')
    axes[0].plot(y_pred_LR, '-', label='Linear Regression')
    axes[0].legend(loc='upper right')
    axes[0].set_title(f'Linear Regression Prediction Trend')

    axes[1].plot(y_test_sorted, '-', label='Raw Data')
    axes[1].plot(y_pred_XGB, '-', label='XGBoost Regressor')
    axes[1].legend(loc='upper right')
    axes[1].set_title(f'XGBoost Regressor Prediction Trend')

    axes[2].plot(y_test_sorted, '-', label='Raw Data')
    axes[2].plot(y_pred_RF, '-', label='Random Forest')
    axes[2].legend(loc='upper right')
    axes[2].set_title(f'Random Forest Prediction Trend')

def calculate_errors(y_true, y_pred):
    error_scores={}
    error_scores['mse'] = mean_squared_error(y_true, y_pred)
    error_scores['mae'] = mean_absolute_error(y_true, y_pred)
    error_scores['rmse'] = np.sqrt(error_scores['mse'])
    error_scores['r2'] = r2_score(y_true, y_pred)
    return error_scores

def model_errors(data, y_test):
    model_errors={}
    for key, y_pred in data.items():
        model_errors[key] = calculate_errors(y_test,y_pred)
    return model_errors

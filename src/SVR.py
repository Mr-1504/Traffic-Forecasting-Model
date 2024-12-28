import json
import os

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import matplotlib.pyplot as plt

file_path = '../resource/train27303.csv'
metrics_path = '../res/SVR/svr_metrics.json'
model_path = '../res/SVR/svr_model.h5'
time_step = 24


def load_best_metrics():
    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            print(f"[Load model successfully]")
            return metrics
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[Error loading metrics: {e}]")
    print("[Not found model or invalid metrics]")
    return None


def save_model_and_metrics(model, r2, mse, rmse):
    joblib.dump(model, model_path)
    metrics_data = {
        "r2_score": r2,
        "mse": mse,
        "rmse": rmse
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=3)

    print(f'\n[Mô hình đã lưu], R²: {r2:.4f}, File: {model_path}')


def plot(test_data, y_test, test_predict, model=None):
    mse = mean_squared_error(y_test, test_predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, test_predict)

    best_metrics = load_best_metrics()
    if best_metrics is None or r2 > best_metrics['r2_score']:
        save_model_and_metrics(model, r2, mse, rmse)

    print(f'MSE: {mse:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'R²: {r2:.4f}')

    if 'timestamp' not in test_data.columns:
        raise ValueError("[Test data thiếu cột 'timestamp']")

    plt.figure(figsize=(20, 10))
    plt.plot(test_data['timestamp'][:len(y_test)], y_test, label='Real Traffic Count', color='red')
    plt.plot(test_data['timestamp'][:len(test_predict)], test_predict, label='Predicted Traffic Count', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Traffic Count')
    plt.title('Traffic Prediction')
    plt.legend()
    plt.savefig(os.path.join('..', 'res', 'SVR', 'svr.png'))
    plt.close()


def read_data():
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.sort_values('timestamp', inplace=True)

    train_data = data[data['timestamp'] < '2015-12-27']
    test_data = data[(data['timestamp'] >= '2015-12-27') & (data['timestamp'] <= '2015-12-30')]

    train_traffic = train_data['hourly_traffic_count'].values.reshape(-1, 1)
    test_traffic = test_data['hourly_traffic_count'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_traffic)
    test_scaled = scaler.transform(test_traffic)

    return test_data, train_scaled, test_scaled, scaler


def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)


def train():
    test_data, train_scaled, test_scaled, scaler = read_data()
    x_train, y_train = create_dataset(train_scaled, time_step)
    x_test, y_test = create_dataset(test_scaled, time_step)

    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(x_train, y_train)

    test_predict = model.predict(x_test)

    test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    return test_data, y_test, test_predict, model


if __name__ == '__main__':
    test_data, y_test, test_predict, model = train()
    plot(test_data, y_test, test_predict, model)

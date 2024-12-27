# LSTM
import json
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import Callback

file_path = '../resource/train27303.csv'
metrics_path = '../res/LSTM/lstm_metrics.json'
time_step = 24
train_epoch = 200
batch_size = 32


class CustomSaveCallback(Callback):
    def __init__(self, x_test, y_test, scaler, best_r2):
        super().__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.scaler = scaler
        self.best_r2 = best_r2
        self.best_predictions = None

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.x_test)
        predictions = self.scaler.inverse_transform(predictions)
        y_true = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

        r2 = r2_score(y_true, predictions)

        if r2 > self.best_r2:
            self.best_r2 = r2
            self.best_predictions = predictions
            model_path = f'../res/LSTM/lstm_model.h5'
            self.model.save(model_path)

            metrics_data = {
                "epoch": epoch + 1,
                "r2_score": r2,
                "mse": mean_squared_error(y_true, predictions),
                "rmse": np.sqrt(mean_squared_error(y_true, predictions))
            }
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=4)

            print(f'\n[Mô hình đã lưu], R²: {r2:.4f}, File: {model_path}')


def load_best_metrics():
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        print(f"[Load model seccussfully]")
        print(f"Epoch: {metrics['epoch']}")
        print(f"R²: {metrics['r2_score']:.4f}")
        print(f"MSE: {metrics['mse']:.2f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        return metrics
    else:
        print("[Not found model]")
        return None


def plot(test_data, y_test, best_predictions):
    if best_predictions is None:
        print('No model to plot')
        return
    mse = mean_squared_error(y_test, best_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, best_predictions)

    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R²: {r2}')

    plt.figure(figsize=(20, 10))

    if len(test_data['timestamp']) > len(y_test):
        test_data = test_data.iloc[:len(y_test)]

    plt.plot(test_data['timestamp'], y_test, label='Real Traffic Count', color='red')
    plt.plot(test_data['timestamp'], best_predictions, label='Predicted Traffic Count', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Traffic Count')
    plt.title('Traffic Prediction')
    plt.legend()
    plt.savefig('../res/LSTM/lstm.png')
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

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    best_metrics = load_best_metrics()
    save_callback = CustomSaveCallback(x_test, y_test, scaler, best_metrics['r2_score'] if best_metrics else -np.inf)
    model.fit(
        x_train,
        y_train,
        epochs=train_epoch,
        batch_size=batch_size,
        verbose=1,
        callbacks=save_callback
    )

    return test_data, save_callback.scaler.inverse_transform(
        save_callback.y_test.reshape(-1, 1)), save_callback.best_predictions


if __name__ == '__main__':
    test_data, y_test, test_predict = train()
    plot(test_data, y_test, test_predict)

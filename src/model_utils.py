import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import Callback


class CustomSaveCallback(Callback):
    def __init__(self, x_test, y_test, scaler, best_r2, model_path, metrics_path):
        super().__init__()
        self.model_path = model_path
        self.metrics_path = metrics_path
        self.x_test = x_test
        self.y_test = y_test
        self.scaler = scaler
        self.best_r2 = best_r2
        self.best_predictions = None

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.x_test)
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))
        y_true = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

        r2 = r2_score(y_true, predictions)
        print(f'\nEpoch: {epoch + 1}, R²: {r2:.4f}, Best R²: {self.best_r2:.4f}')

        if r2 > self.best_r2:
            self.best_r2 = r2
            self.best_predictions = predictions.copy()
            self.model.save(self.model_path)
            with open(self.metrics_path, 'w') as f:
                json.dump({
                    "epoch": epoch + 1,
                    "r2_score": r2,
                    "mse": mean_squared_error(y_true, predictions),
                    "rmse": np.sqrt(mean_squared_error(y_true, predictions))
                }, f, indent=4)
            print(f"\n[Model Saved] Epoch: {epoch + 1}, R2: {r2:.4f}")


def load_best_metrics(metrics_path):
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

def read_data(file_path):
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    scaler = MinMaxScaler()
    data['scaled_count'] = scaler.fit_transform(data['hourly_traffic_count'].values.reshape(-1, 1))

    train_data = data[data['timestamp'] < '2015-12-27']
    test_data = data[data['timestamp'] >= '2015-12-27']
    return train_data, test_data, scaler


def create_dataset(data, time_step):
    sequences, targets = [], []
    for i in range(len(data) - time_step,):
        seq = data[i:i + time_step]
        sequences.append(seq)
        targets.append(data[i + time_step])
    return np.array(sequences), np.array(targets)


def plot_results(test_data, y_test, predictions, scaler, plot_path, metrics_path):
    if predictions is None:
        print('No model to plot')
        return
    with open(metrics_path, 'r') as f:
        best_metrics = json.load(f)
    print(f"R2: {best_metrics['r2_score']:.4f}\nMSE: {best_metrics['mse']:.2f}\nRMSE: {best_metrics['rmse']:.2f}")
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.figure(figsize=(20, 10))
    plt.plot(test_data['timestamp'][:len(y_test)], y_test, label='True Values', color='red')
    plt.plot(test_data['timestamp'][:len(predictions)], predictions, label='Predictions', color='blue')
    plt.xlabel('Timestamp')
    plt.ylabel('Traffic Count')
    plt.title('Traffic Prediction')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()
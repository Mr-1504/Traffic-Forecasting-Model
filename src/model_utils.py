import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import Callback


class CustomSaveCallback(Callback):
    def __init__(self, x_test, y_test, scaler, args):
        best_metrics = load_best_metrics(args.metrics_path)
        super().__init__()
        self.model_path = args.model_path
        self.metrics_path = args.metrics_path
        self.x_test = x_test
        self.y_test = y_test
        self.scaler = scaler
        self.best_r2 = best_metrics['r2_score'] if best_metrics else -np.inf
        self.best_mae = best_metrics['mae'] if best_metrics else np.inf
        self.best_predictions = None

    def on_epoch_end(self, epoch, logs=None):
        test_predictions = self.model.predict(self.x_test)
        test_predictions = self.scaler.inverse_transform(test_predictions.reshape(-1, 1))
        y_test = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        test_r2 = r2_score(y_test, test_predictions)
        mae = mean_absolute_error(y_test, test_predictions)
        print(f'\nEpoch: {epoch + 1}, R²: {test_r2:.4f}, MAE: {mae:.4f}, Best MAE: {self.best_mae:.4f}')

        if mae < self.best_mae:
            self.best_r2 = test_r2
            self.best_predictions = test_predictions.copy()
            self.best_mae = mae
            self.model.save(self.model_path)
            with open(self.metrics_path, 'w') as f:
                json.dump({
                    "epoch": epoch + 1,
                    "r2_score": test_r2,
                    "mae": mae,
                    "mse": mean_squared_error(y_test, test_predictions),
                    "rmse": np.sqrt(mean_squared_error(y_test, test_predictions))
                }, f, indent=5)
            print(f"\n[Model Saved] Epoch: {epoch + 1}, MAE: {mae:.4f}, R2: {test_r2:.4f}")


def is_valid_metrics(metrics_path):
    if os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics.get('r2_score') is not None and metrics.get('mae') is not None and metrics.get(
            'mse') is not None and metrics.get('rmse') is not None and metrics.get('epoch') is not None
    return False


def load_best_metrics(metrics_path):
    if is_valid_metrics(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            print(f"[Load model successfully]")
            print(f"Epoch: {metrics['epoch']}")
            print(f"R²: {metrics['r2_score']:.4f}")
            print(f"MAE: {metrics['mae']:.2f}")
            print(f"MSE: {metrics['mse']:.2f}")
            print(f"MAPE: {metrics['mape']:.2f}")
            print(f"RMSE: {metrics['rmse']:.2f}")
            return metrics
        except json.JSONDecodeError:
            print("[Error] Invalid JSON file")
            return None
    else:
        print("[Not found model]")
        return None


def read_data(file_path):
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    data['day'] = data['timestamp'].dt.day
    data['weekday'] = data['timestamp'].dt.weekday

    scaler = MinMaxScaler()
    data['scaled_count'] = scaler.fit_transform(data['hourly_traffic_count'].values.reshape(-1, 1))

    train_data = data[data['timestamp'] <= '2015-12-25']
    test_data = data[((data['timestamp'] > '2015-12-22') & (data['timestamp'] <= '2015-12-31'))]
    return train_data, test_data, scaler


def create_dataset(data, time_step):
    sequences, targets = [], []
    for i in range(len(data) - time_step, ):
        seq = data[i:i + time_step]
        sequences.append(seq)
        targets.append(data[i + time_step])
    return np.array(sequences), np.array(targets)


def plot_results(test_data, callback, args):
    if callback.best_predictions is None:
        print('No model to plot')
        return

    with open(args.metrics_path, 'r') as f:
        best_metrics = json.load(f)

    print(
        f"R2: {best_metrics['r2_score']:.4f}\nMSE: {best_metrics['mse']:.2f}\nMAE: {best_metrics['mae']:.2f}\nRMSE: {best_metrics['rmse']:.2f}")
    y_test = callback.scaler.inverse_transform(callback.y_test.reshape(-1, 1))

    plt.figure(figsize=(20, 10))
    plt.plot(test_data['timestamp'][:len(y_test)], y_test, label='True Values', color='red')
    plt.plot(test_data['timestamp'][:len(callback.best_predictions)], callback.best_predictions, label='Predictions',
             color='blue')
    plt.xlabel('Timestamp')
    plt.ylabel('Traffic Count')
    plt.title('Traffic Prediction')
    plt.legend()
    plt.savefig(args.plot_path)
    plt.close()

import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import Callback


class CustomSaveCallback(Callback):
    def __init__(self, x_test, y_test, scaler, best_r2, args):
        super().__init__()
        self.model_path = args.model_path
        self.metrics_path = args.metrics_path
        self.x_test = x_test
        self.y_test = y_test
        self.scaler = scaler
        self.best_r2 = best_r2
        self.best_predictions = None

    def on_epoch_end(self, epoch, logs=None):
        test_predictions = self.model.predict(self.x_test)
        print("test_predictions.shape", test_predictions.shape)
        print("y_test.shape", self.y_test.shape)
        # test_predictions = test_predictions[:, 1,:]
        test_predictions = self.scaler.inverse_transform(test_predictions.reshape(-1, 1))
        y_test = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

        test_r2 = r2_score(y_test, test_predictions)

        print(f'\nEpoch: {epoch + 1}, R²: {test_r2:.4f}, Best R²: {self.best_r2:.4f}')

        if test_r2 > self.best_r2:
            self.best_r2 = test_r2
            self.best_predictions = test_predictions.copy()
            self.model.save(self.model_path)
            with open(self.metrics_path, 'w') as f:
                json.dump({
                    "epoch": epoch + 1,
                    "r2_score": test_r2,
                    "mse": mean_squared_error(y_test, test_predictions),
                    "rmse": np.sqrt(mean_squared_error(y_test, test_predictions))
                }, f, indent=4)
            print(f"\n[Model Saved] Epoch: {epoch + 1}, R2: {test_r2:.4f}")


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
    data['hour'] = data['timestamp'].dt.hour
    data['day'] = data['timestamp'].dt.day
    data['weekday'] = data['timestamp'].dt.weekday

    scaler = MinMaxScaler()
    data['scaled_count'] = scaler.fit_transform(data['hourly_traffic_count'].values.reshape(-1, 1))

    train_data = data[data['timestamp'] < '2015-12-27']
    test_data = data[data['timestamp'] >= '2015-12-27']
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

    print(f"R2: {best_metrics['r2_score']:.4f}\nMSE: {best_metrics['mse']:.2f}\nRMSE: {best_metrics['rmse']:.2f}")
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

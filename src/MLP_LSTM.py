from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, concatenate
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt

file_path = '../resource/train27303.csv'
metrics_path = '../res/MLP_LSTM/combined_metrics.json'
time_step = 12
train_epoch = 200
batch_size = 64

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
            model_path = f'../res/MLP_LSTM/combined_model.h5'
            self.model.save(model_path)

            metrics_data = {
                "epoch": epoch + 1,
                "r2_score": r2,
                "mse": mean_squared_error(y_true, predictions),
                "rmse": np.sqrt(mean_squared_error(y_true, predictions))
            }
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=4)

            print(f'\n[Model Saved], R²: {r2:.4f}, File: {model_path}')

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

def create_combined_model(time_step):
    lstm_input = Input(shape=(time_step, 1))
    lstm_branch = LSTM(50, return_sequences=True)(lstm_input)
    lstm_branch = LSTM(50)(lstm_branch)

    mlp_input = Input(shape=(time_step,))
    mlp_branch = Dense(128, activation='tanh')(mlp_input)
    mlp_branch = Dense(64, activation='tanh')(mlp_branch)

    combined = concatenate([lstm_branch, mlp_branch])
    combined = Dense(32, activation='relu')(combined)
    output = Dense(1)(combined)

    model = Model(inputs=[lstm_input, mlp_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train():
    test_data, train_scaled, test_scaled, scaler = read_data()
    x_train_lstm, y_train = create_dataset(train_scaled, time_step)
    x_test_lstm, y_test = create_dataset(test_scaled, time_step)

    x_train_mlp = x_train_lstm.reshape(x_train_lstm.shape[0], x_train_lstm.shape[1])
    x_test_mlp = x_test_lstm.reshape(x_test_lstm.shape[0], x_test_lstm.shape[1])

    x_train_lstm = x_train_lstm.reshape(x_train_lstm.shape[0], x_train_lstm.shape[1], 1)
    x_test_lstm = x_test_lstm.reshape(x_test_lstm.shape[0], x_test_lstm.shape[1], 1)

    model = create_combined_model(time_step)
    best_metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            best_metrics = json.load(f)
    save_callback = CustomSaveCallback([x_test_lstm, x_test_mlp], y_test, scaler,
                                        best_metrics['r2_score'] if best_metrics else -np.inf)

    model.fit(
        [x_train_lstm, x_train_mlp],
        y_train,
        epochs=train_epoch,
        batch_size=batch_size,
        verbose=1,
        callbacks=[save_callback]
    )
    return test_data, scaler.inverse_transform(y_test.reshape(-1, 1)), save_callback.best_predictions

def plot(test_data, y_test, best_predictions):
    if best_predictions is None:
        print('No predictions to plot')
        return
    plt.figure(figsize=(20, 10))
    plt.plot(test_data['timestamp'], y_test, label='Real Traffic Count', color='red')
    plt.plot(test_data['timestamp'], best_predictions, label='Predicted Traffic Count', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Traffic Count')
    plt.title('MLP_LSTM Model Prediction')
    plt.legend()
    plt.savefig('../res/MLP_LSTM/mpl_lstm.png')
    plt.close()

if __name__ == '__main__':
    test_data, y_test, test_predict = train()
    plot(test_data, y_test, test_predict)

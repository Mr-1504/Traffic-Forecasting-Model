import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from src.model_utils import read_data, create_dataset, CustomSaveCallback, load_best_metrics, plot_results

file_path = '../resource/train27303.csv'
metrics_path = '../res/LSTM/lstm_metrics.json'
plot_path = '../res/LSTM/lstm_plot.png'
model_path = '../res/LSTM/lstm_model.h5'
time_step = 64
train_epoch = 200
batch_size = 64


def train():
    train_data, test_data, scaler = read_data(file_path)

    x_train, y_train = create_dataset(train_data['scaled_count'].values, time_step)
    x_test, y_test = create_dataset(test_data['scaled_count'].values, time_step)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    best_metrics = load_best_metrics(metrics_path)
    callback = CustomSaveCallback(x_test, y_test, scaler,
                                  best_metrics['r2_score'] if best_metrics else -np.inf, model_path, metrics_path)

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(time_step, 1)),
        Dropout(0.2),
        BatchNormalization(),

        LSTM(64, return_sequences=False),
        Dropout(0.2),

        Dense(64, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    model.fit(
        x_train, y_train,
        epochs=train_epoch,
        batch_size=batch_size,
        callbacks=[callback],
        verbose=1
    )

    return test_data, y_test, scaler, callback.best_predictions


if __name__ == '__main__':
    test_data, y_test, scaler, best_predictions = train()
    plot_results(test_data, y_test, best_predictions, scaler, plot_path, metrics_path)

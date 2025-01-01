# GRU
import numpy as np
from keras.src.layers import GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from src.model_utils import read_data, create_dataset, load_best_metrics, CustomSaveCallback, plot_results

file_path = '../resource/train27303.csv'
metrics_path = '../res/GRU/gru_metrics.json'
plot_path = '../res/GRU/gru_plot.png'
model_path = '../res/GRU/gru_model.h5'
time_step = 12
train_epoch = 100
batch_size = 64
training_rate = 0.8


def train():
    train_data, test_data, scaler = read_data(file_path)

    x_train, y_train = create_dataset(train_data['scaled_count'].values, time_step)
    x_test, y_test = create_dataset(test_data['scaled_count'].values, time_step)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(GRU(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    best_metrics = load_best_metrics(metrics_path)
    save_callback = CustomSaveCallback(x_test, y_test, scaler, best_metrics['r2_score'] if best_metrics else -np.inf,
                                       model_path, metrics_path)
    model.fit(
        x_train,
        y_train,
        epochs=train_epoch,
        batch_size=batch_size,
        verbose=1,
        callbacks=save_callback
    )

    return test_data, y_test, scaler, save_callback.best_predictions

if __name__ == '__main__':
    test_data, y_test, scaler, best_predictions = train()
    plot_results(test_data, y_test, best_predictions, scaler, plot_path, metrics_path)

import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from src.model_utils import read_data, create_dataset, CustomSaveCallback, load_best_metrics, plot_results

file_path = '../resource/train27303.csv'
metrics_path = '../res/MLP/mlp_metrics.json'
plot_path = '../res/MLP/mlp_plot.png'
model_path = '../res/MLP/mlp_model.h5'
time_step = 24
train_epoch = 500
batch_size = 64


def train():
    train_data, test_data, scaler = read_data(file_path)

    x_train, y_train = create_dataset(train_data['scaled_count'].values, time_step)
    x_test, y_test = create_dataset(test_data['scaled_count'].values, time_step)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    model = Sequential()
    model.add(Dense(128, activation='tanh', input_shape=(time_step,1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(64, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='tanh'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mean_squared_error')

    best_metrics = load_best_metrics(metrics_path)
    save_callback = CustomSaveCallback(x_test, y_test, scaler, best_metrics['r2_score'] if best_metrics else -np.inf,
                                       model_path, metrics_path)
    model.fit(
        x_train, y_train,
        epochs=train_epoch,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=1,
        callbacks=save_callback
    )

    return test_data, y_test, scaler, save_callback.best_predictions


if __name__ == '__main__':
    test_data, test_targets, scaler, best_predictions = train()
    plot_results(test_data, test_targets, best_predictions, scaler, plot_path, metrics_path)

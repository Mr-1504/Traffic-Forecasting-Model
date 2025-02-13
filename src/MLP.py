import argparse

import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from src.model_utils import read_data, create_dataset, CustomSaveCallback, load_best_metrics, plot_results

# file_path = '../resource/train27303.csv'
# metrics_path = '../res/MLP/mlp_metrics.json'
# plot_path = '../res/MLP/mlp_plot.png'
# model_path = '../res/MLP/mlp_model.h5'
# time_step = 24
# train_epoch = 500
# batch_size = 64


def train(args):
    train_data, test_data, scaler = read_data(args.file_path)

    x_train, y_train = create_dataset(train_data['scaled_count'].values, args.time_step)
    x_test, y_test = create_dataset(test_data['scaled_count'].values, args.time_step)

    print("x_train shape: ", x_train.shape)
    print("x_test shape: ", x_test.shape)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])

    print("x_train shape: ", x_train.shape)
    print("x_test shape: ", x_test.shape)

    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(args.time_step,)))

    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mae')

    best_metrics = load_best_metrics(args.metrics_path)
    save_callback = CustomSaveCallback(x_test, y_test, scaler, best_metrics['r2_score'] if best_metrics else -np.inf,
                                       args)
    model.fit(
        x_train, y_train,
        epochs=args.train_epoch,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=save_callback
    )

    return test_data, save_callback


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default="../resource/train27303.csv")
    parser.add_argument('--metrics_path', type=str, default="../res/MLP/mlp_metrics.json")
    parser.add_argument('--plot_path', type=str, default="../res/MLP/mlp_plot.png")
    parser.add_argument('--model_path', type=str, default="../res/MLP/mlp_model.h5")
    parser.add_argument('--checkpoint_path', type=str, default="../res/MLP/mlp_checkpoint.keras")
    parser.add_argument('--time_step', type=int, default=64)
    parser.add_argument('--train_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)

    args, unknown = parser.parse_known_args()
    test_data, call_back = train(args)
    plot_results(test_data, call_back, args)

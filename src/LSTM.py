import argparse

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from src.model_utils import read_data, create_dataset, CustomSaveCallback, load_best_metrics, plot_results

def train(args):
    train_data, test_data, scaler = read_data(args.file_path)

    x_train, y_train = create_dataset(train_data['scaled_count'].values, args.time_step)
    x_test, y_test = create_dataset(test_data['scaled_count'].values, args.time_step)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    best_metrics = load_best_metrics(args.metrics_path)
    callback = CustomSaveCallback(x_test, y_test, scaler,
                                  best_metrics['r2_score'] if best_metrics else -np.inf, args)

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(args.time_step, 1)),
        Dropout(0.3),
        BatchNormalization(),

        LSTM(64, return_sequences=True),
        Dropout(0.3),

        LSTM(32),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mae')
    checkpoint = ModelCheckpoint(args.checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

    model.fit(
        x_train, y_train,
        epochs=args.train_epoch,
        batch_size=args.batch_size,
        callbacks=[callback, checkpoint],
        verbose=1
    )

    return test_data, callback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default="../resource/train27303.csv")
    parser.add_argument('--metrics_path', type=str, default="../res/LSTM/lstm_metrics.json")
    parser.add_argument('--plot_path', type=str, default="../res/LSTM/lstm_plot.png")
    parser.add_argument('--model_path', type=str, default="../res/LSTM/lstm_model.h5")
    parser.add_argument('--checkpoint_path', type=str, default="../res/LSTM/lstm_checkpoint.keras")
    parser.add_argument('--time_step', type=int, default=64)
    parser.add_argument('--train_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)

    args, unknown = parser.parse_known_args()
    test_data, callback = train(args)
    plot_results(test_data, callback, args)

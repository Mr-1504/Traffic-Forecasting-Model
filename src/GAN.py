import json
import os

import pandas as pd
import numpy as np
from keras.src.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import Callback

file_path = '../resource/train27303.csv'
metrics_path = '../res/GAN/gan_metrics.json'
time_step = 24
train_epoch = 100
batch_size = 32
latent_dim = 100


class CustomSaveCallback(Callback):
    def __init__(self, x_test, y_test, scaler, best_r2, generator):
        super().__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.scaler = scaler
        self.best_r2 = best_r2
        self.best_predictions = None
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        noise = np.random.normal(0, 1, (len(self.y_test), latent_dim))
        test_predict = self.generator.predict(noise)

        if test_predict.shape[1] != time_step:
            test_predict = test_predict.reshape(-1, time_step)
        predictions = self.model.predict(test_predict)
        predictions = self.scaler.inverse_transform(predictions)
        y_true = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

        r2 = r2_score(y_true, predictions)

        if r2 > self.best_r2:
            self.best_r2 = r2
            self.best_predictions = predictions
            model_path = f'../res/GAN/gan_model.h5'
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
        print(f"R²: {metrics['r2_score']:.2f}")
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
    plt.savefig('../res/GAN/GAN.png')
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
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)


def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(784, activation='sigmoid'))
    return model


def build_discriminator():
    model = Sequential()
    model.add(Dense(512, input_dim=784))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model


def train_gan():
    test_data, train_scaled, test_scaled, scaler = read_data()
    x_train, y_train = create_dataset(train_scaled, time_step)
    x_test, y_test = create_dataset(test_scaled, time_step)

    generator = build_generator()
    discriminator = build_discriminator()

    optimizer = Adam(learning_rate=0.0001)

    discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer=optimizer, loss='binary_crossentropy')

    best_metrics = load_best_metrics()
    save_callback = CustomSaveCallback(x_test, y_test, scaler, best_metrics['r2_score'] if best_metrics else -np.inf,
                                       generator)

    for epoch in range(train_epoch):
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_traffic = generator.predict(noise)

        real_traffic = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_traffic, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_traffic, fake_labels)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_y)

        if epoch % 10 == 0:
            print(
                f"{epoch} [D loss real: {d_loss_real[0]}, acc.: {d_loss_real[1]}] [D loss fake: {d_loss_fake[0]}, acc.: {d_loss_fake[1]}] [G loss: {g_loss}]")

        save_callback.on_epoch_end(epoch)

    return test_data, save_callback.scaler.inverse_transform(
        save_callback.y_test.reshape(-1, 1)), save_callback.best_predictions


if __name__ == '__main__':
    test_data, y_test, test_predict = train_gan()
    plot(test_data, y_test, test_predict)

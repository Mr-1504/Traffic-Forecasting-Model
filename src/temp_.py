# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# import tensorflow as tf
#
# # Load dữ liệu
# data = pd.read_csv('../resource/train27303.csv')
# data['timestamp'] = pd.to_datetime(data['timestamp'])
# data.set_index('timestamp', inplace=True)
#
# # Chuẩn hóa dữ liệu
# scaler = MinMaxScaler(feature_range=(0, 1))
# data['normalized_traffic'] = scaler.fit_transform(data[['hourly_traffic_count']])
#
#
# # Hàm tạo sequences
# def create_sequences(data, sequence_length):
#     X, y = [], []
#     for i in range(len(data) - sequence_length):
#         X.append(data[i:i + sequence_length])
#         y.append(data[i + sequence_length])
#     return np.array(X), np.array(y)
#
#
# # Tạo dữ liệu train/test
# sequence_length = 30
# normalized_traffic = data['normalized_traffic'].values
# X, y = create_sequences(normalized_traffic, sequence_length)
#
# train_size = int(0.8 * len(X))
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]
#
# # Xây dựng mô hình LSTM
# lstm_model = Sequential([
#     LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=True),
#     Dropout(0.2),
#     LSTM(64, return_sequences=False),
#     Dropout(0.2),
#     Dense(1, activation='linear')
# ])
#
# # Compile mô hình
# lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
#
# # Huấn luyện
# history = lstm_model.fit(
#     X_train[..., np.newaxis], y_train,
#     validation_data=(X_test[..., np.newaxis], y_test),
#     epochs=10, batch_size=64, verbose=1
# )
#
# # Lưu mô hình
# lstm_model.save('lstm_traffic_model.h5')
#
#
# # GAN để sinh dữ liệu
#
# def build_generator(input_dim, output_dim):
#     model = Sequential([
#         Dense(128, activation='relu', input_dim=input_dim),
#         Dense(256, activation='relu'),
#         Dense(output_dim, activation='tanh')
#     ])
#     return model
#
#
# def build_discriminator(input_dim):
#     model = Sequential([
#         Dense(256, activation='relu', input_dim=input_dim),
#         Dense(128, activation='relu'),
#         Dense(1, activation='sigmoid')
#     ])
#     return model
#
#
# # Tham số cho GAN
# z_dim = 30  # Kích thước noise vector
# generator = build_generator(z_dim, sequence_length)
# discriminator = build_discriminator(sequence_length)
#
# # Compile discriminator
# discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # GAN kết hợp generator và discriminator
# discriminator.trainable = False
# z = tf.keras.Input(shape=(z_dim,))
# generated_data = generator(z)
# validity = discriminator(generated_data)
# combined = tf.keras.Model(z, validity)
# combined.compile(optimizer='adam', loss='binary_crossentropy')
#
# # Huấn luyện GAN
# batch_size = 64
# epochs = 100
#
# for epoch in range(epochs):
#     # Huấn luyện discriminator
#     idx = np.random.randint(0, X_train.shape[0], batch_size)
#     real_data = X_train[idx]
#     noise = np.random.normal(0, 1, (batch_size, z_dim))
#     fake_data = generator.predict(noise)
#
#     d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
#     d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
#     d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
#
#     # Huấn luyện generator
#     noise = np.random.normal(0, 1, (batch_size, z_dim))
#     g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))
#
#     # In tiến trình
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss[0]} | G Loss: {g_loss}")
#
# # Sinh dữ liệu kiểm thử
# noise = np.random.normal(0, 1, (100, z_dim))
# generated_data = generator.predict(noise)
#
# # Dự đoán bằng LSTM và đánh giá
# predicted = lstm_model.predict(generated_data[..., np.newaxis])
#
# # Đảo ngược chuẩn hóa
# generated_data_rescaled = scaler.inverse_transform(generated_data)
# predicted_rescaled = scaler.inverse_transform(predicted)
#
# # Tính toán chỉ số đánh giá
# mse = mean_squared_error(
#     generated_data_rescaled[:, -1],
#     predicted_rescaled[:, 0]
# )
# rmse = np.sqrt(mse)
# r2 = r2_score(
#     generated_data_rescaled[:, -1],
#     predicted_rescaled[:, 0]
# )
# print(f"MSE: {mse}")
# print(f"RMSE: {rmse}")
# print(f"R2 Score: {r2}")
#
# plt.figure(figsize=(14, 7))
#
# # Biểu đồ dữ liệu thực
# plt.plot(generated_data_rescaled, label='Real Data', linestyle='--', color='blue')
#
# # Biểu đồ dự đoán
# plt.plot(predicted_rescaled, label='Predicted Data', linestyle='-', color='orange')
#
# # Thêm tiêu đề, nhãn và chú thích
# plt.title('Comparison of Real and Predicted Data (Full Dataset)', fontsize=16)
# plt.xlabel('Index', fontsize=14)
# plt.ylabel('Traffic Count', fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True)
#
# # Hiển thị biểu đồ
# plt.savefig('../resource/temp_.png')
# plt.close()
# # Lưu dữ liệu sinh ra
# np.save('../resource/generated_data.npy', generated_data)

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Input, LeakyReLU, Reshape, Flatten
from tensorflow.keras.optimizers import Adam

from src.model_utils import read_data, create_dataset, CustomSaveCallback, load_best_metrics, plot_results


def build_generator(latent_dim, time_step):
    model = Sequential([
        Dense(256, input_dim=latent_dim),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dense(512),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dense(time_step, activation='sigmoid'),
        Reshape((time_step, 1))
    ])
    return model

def build_discriminator(time_step):
    model = Sequential([
        Input(shape=(time_step, 1)),
        Flatten(),
        Dense(512),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

def train_gan(x_train, latent_dim, time_step, epochs, batch_size):
    generator = build_generator(latent_dim, time_step)
    discriminator = build_discriminator(time_step)
    discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_data = generator.predict(noise, verbose=0)
        real_data = x_train[np.random.randint(0, x_train.shape[0], batch_size)]

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {0.5 * (d_loss_real[0] + d_loss_fake[0])}] [G loss: {g_loss}]")

    return generator


def train(args):
    train_data, test_data, scaler = read_data(args.file_path)

    x_train, y_train = create_dataset(train_data['scaled_count'].values, args.time_step)
    x_test, y_test = create_dataset(test_data['scaled_count'].values, args.time_step)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    generator = train_gan(x_train, 100, args.time_step, 1000, args.batch_size)

    # Sinh dữ liệu giả từ GAN
    noise = np.random.normal(0, 1, (1000, 100))
    synthetic_data = generator.predict(noise, verbose=0)
    synthetic_labels = np.random.choice(y_train, size=1000)  # Gán nhãn ngẫu nhiên

    # Xuất dữ liệu GAN sinh ra vào file CSV
    synthetic_data_flattened = synthetic_data.flatten()  # Chuyển dữ liệu thành 1 chiều
    df_synthetic_data = pd.DataFrame(synthetic_data_flattened, columns=["Generated Data"])
    df_synthetic_data.to_csv(args.output_file_path, index=False)

    # Vẽ biểu đồ dữ liệu thật và dữ liệu giả
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(y_train)), y_train, label='Real Data', color='b')
    plt.plot(np.arange(len(synthetic_data_flattened)), synthetic_data_flattened, label='Generated Data', color='r', alpha=0.7)
    plt.legend()
    plt.xlabel("Index")
    plt.ylabel("Data Value")
    plt.title("Real vs Generated Data")
    plt.savefig(args.plot_path)  # Lưu biểu đồ vào file
    plt.show()

    best_metrics = load_best_metrics(args.metrics_path)
    callback = CustomSaveCallback(synthetic_data, synthetic_labels, scaler,
                                  best_metrics['r2_score'] if best_metrics else -np.inf, args)

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(args.time_step, 1)),
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
        epochs=args.train_epoch,
        validation_data=(x_test, y_test),
        batch_size=args.batch_size,
        callbacks=[callback],
        verbose=1
    )

    return test_data, callback,


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default="../resource/train27303.csv")
    parser.add_argument('--metrics_path', type=str, default="../res/LSTM/lstm_metrics.json")
    parser.add_argument('--plot_path', type=str, default="../res/LSTM/lstm_plot.png")
    parser.add_argument('--model_path', type=str, default="../res/LSTM/lstm_model.h5")
    parser.add_argument('--time_step', type=int, default=64)
    parser.add_argument('--train_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_file_path', type=str, default="../res/LSTM/generated_data.csv")

    args = parser.parse_args()
    test_data, callback = train(args)
    plot_results(test_data, callback, args)

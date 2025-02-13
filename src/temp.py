import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, LeakyReLU, Input, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from src.model_utils import read_data, create_dataset, plot_results

# GAN
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

# LSTM
def build_lstm_model(time_step):
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
    return model

# Huấn luyện GAN
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

# Huấn luyện LSTM
def train_lstm(x_train, y_train, time_step, epochs, batch_size):
    model = build_lstm_model(time_step)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

if __name__ == "__main__":
    file_path = "../resource/train27303.csv"
    time_step = 64
    latent_dim = 100
    gan_epochs = 1000
    lstm_epochs = 50
    batch_size = 64

    # Đọc dữ liệu
    train_data, _, scaler = read_data(file_path)
    x_train, y_train = create_dataset(train_data['scaled_count'].values, time_step)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    # Huấn luyện GAN
    generator = train_gan(x_train, latent_dim, time_step, gan_epochs, batch_size)

    # Sinh dữ liệu giả từ GAN
    noise = np.random.normal(0, 1, (1000, latent_dim))
    synthetic_data = generator.predict(noise, verbose=0)
    synthetic_labels = np.random.choice(y_train, size=1000)  # Gán nhãn ngẫu nhiên

    # Kết hợp dữ liệu thật và giả
    x_combined = np.concatenate([x_train, synthetic_data])
    y_combined = np.concatenate([y_train, synthetic_labels])

    # Huấn luyện LSTM
    lstm_model = train_lstm(x_combined, y_combined, time_step, lstm_epochs, batch_size)

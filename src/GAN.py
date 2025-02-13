import argparse

import joblib
import numpy as np
import tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Input, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from src.model_utils import read_data, create_dataset, plot_results
import matplotlib.pyplot as plt

def build_generator(latent_dim, time_step):
    model = Sequential([
        Dense(256, input_dim=latent_dim),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dense(512),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dense(1024),
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

def plot_results(d_losses, g_losses, args):
    plt.figure(figsize=(10, 5))

    # Plot Discriminator Loss
    plt.plot([d[0] for d in d_losses], label="Discriminator Loss")

    # Plot Generator Loss
    plt.plot(g_losses, label="Generator Loss")

    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Save and show plot
    plt.savefig(args.plot_path)
    plt.show()

def display_predictions(generator, scaler, args):
    noise = np.random.normal(0, 1, (10, 100))  # Tạo 10 sample giả
    generated_traffic = generator.predict(noise, verbose=0)  # Dự đoán
    generated_traffic = scaler.inverse_transform(generated_traffic.reshape(-1, args.time_step))  # Đưa về giá trị gốc

    plt.figure(figsize=(12, 6))
    for i, traffic in enumerate(generated_traffic):
        plt.plot(traffic, label=f"Generated {i+1}")
    plt.title("Generated Traffic Predictions")
    plt.xlabel("Time Step")
    plt.ylabel("Traffic Count")
    plt.legend()
    plt.grid(True)
    plt.savefig("../res/GAN/generated_predictions.png")
    plt.show()

def train(args):
    latent_dim = 100
    train_data, test_data, scaler = read_data(args.file_path)
    x_train, y_train = create_dataset(train_data['scaled_count'].values, args.time_step)
    x_test, y_test = create_dataset(test_data['scaled_count'].values, args.time_step)

    generator = build_generator(latent_dim, args.time_step)
    discriminator = build_discriminator(args.time_step)
    discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy',
                          metrics=['accuracy'])

    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

    d_losses, g_losses = [], []

    for epoch in range(args.train_epoch):
        noise = np.random.normal(0, 1, (args.batch_size, latent_dim))
        fake_traffic = generator.predict(noise, verbose=0)
        real_traffic = x_train[np.random.randint(0, x_train.shape[0], args.batch_size)].reshape(args.batch_size, args.time_step, 1)
        real_labels = np.ones((args.batch_size, 1))
        fake_labels = np.zeros((args.batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_traffic, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_traffic, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        d_losses.append(d_loss)

        noise = np.random.normal(0, 1, (args.batch_size, latent_dim))
        valid_y = np.ones((args.batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_y)
        g_losses.append(g_loss)

        if epoch % 10 == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")

    generator.save(args.model_path)
    print(f"Final D loss: {d_losses[-1][0]}, acc.: {100 * d_losses[-1][1]}%")
    print(f"Final G loss: {g_losses[-1]}")
    return d_losses, g_losses, scaler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default="../resource/train27303.csv")
    parser.add_argument('--metrics_path', type=str, default="../res/GAN/gan_metrics.json")
    parser.add_argument('--plot_path', type=str, default="../res/GAN/gan_plot.png")
    parser.add_argument('--model_path', type=str, default="../res/GAN/gan_model.h5")
    parser.add_argument('--time_step', type=int, default=64)
    parser.add_argument('--train_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    d_losses, g_losses, scaler = train(args)
    plot_results(d_losses, g_losses, args)
    generator = tf.keras.models.load_model(args.model_path)
    display_predictions(generator, scaler, args)

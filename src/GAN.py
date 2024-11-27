import pandas as pd
import numpy as np
from keras.src.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

file_path = 'D:/Code/Vigilant-VGG16/Traffic/resource/train27303.csv'
time_step = 24
train_epoch = 100
batch_size = 32
latent_dim = 100  # Kích thước không gian ẩn của Generator

def plot(test_data, y_test, test_predict):
    if test_predict.shape[1] > 1:
        test_predict_sliced = test_predict[:, 0]
    else:
        test_predict_sliced = test_predict.flatten()

    mse = mean_squared_error(y_test, test_predict_sliced)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, test_predict_sliced)

    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R²: {r2}')

    plt.figure(figsize=(20, 10))

    if len(test_data['timestamp']) > len(y_test):
        test_data = test_data.iloc[:len(y_test)]

    plt.plot(test_data['timestamp'], y_test, label='Real Traffic Count', color='red')
    plt.plot(test_data['timestamp'], test_predict_sliced, label='Predicted Traffic Count', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Traffic Count')
    plt.title('Traffic Prediction')
    plt.legend()
    plt.savefig('GAN_Traffic.png')
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
            print(f"{epoch} [D loss real: {d_loss_real[0]}, acc.: {d_loss_real[1]}] [D loss fake: {d_loss_fake[0]}, acc.: {d_loss_fake[1]}] [G loss: {g_loss}]")

    noise = np.random.normal(0, 1, (len(x_test), latent_dim))
    test_predict = generator.predict(noise)

    if test_predict.shape[1] != time_step:
        test_predict = test_predict.reshape(-1, time_step)

    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    return test_data, y_test, test_predict

if __name__ == '__main__':
    test_data, y_test, test_predict = train_gan()
    print("y_test shape:", y_test.shape)
    print("test_predict shape:", test_predict.shape)
    plot(test_data, y_test, test_predict)

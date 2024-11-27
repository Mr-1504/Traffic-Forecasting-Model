# SVR
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import matplotlib.pyplot as plt

file_path = '/train27303.csv'
time_step = 24

def plot(test_data, y_test, test_predict):
    mse = mean_squared_error(y_test, test_predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, test_predict)

    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R²: {r2}')

    plt.figure(figsize=(20, 10))

    if len(test_data['timestamp']) > len(y_test):
        test_data = test_data.iloc[:len(y_test)]

    plt.plot(test_data['timestamp'], y_test, label='Real Traffic Count', color='red')
    plt.plot(test_data['timestamp'], test_predict, label='Predicted Traffic Count', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Traffic Count')
    plt.title('Traffic Prediction')
    plt.legend()
    plt.savefig('SVR.png')
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
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

def train():
    test_data, train_scaled, test_scaled, scaler = read_data()
    x_train, y_train = create_dataset(train_scaled, time_step)
    x_test, y_test = create_dataset(test_scaled, time_step)

    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(x_train, y_train)

    test_predict = model.predict(x_test)

    test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    return test_data, y_test, test_predict

if __name__ == '__main__':
    test_data, y_test, test_predict = train()
    plot(test_data, y_test, test_predict)

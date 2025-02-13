# Cài đặt lại thư viện cần thiết
import numpy as np
from pyswarm import pso  # Thuật toán PSO
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Bidirectional, LSTM

# Import các hàm tiện ích từ project
from src.model_utils import read_data, create_dataset, CustomSaveCallback, load_best_metrics, plot_results

# Đường dẫn file
file_path = '../resource/train27303.csv'
metrics_path = '../res/PSO_BiLSTM/pso_bilstm_metrics.json'
plot_path = '../res/PSO_BiLSTM/pso_bilstm_plot.png'
model_path = '../res/PSO_BiLSTM/pso_bilstm_model.h5'
time_step = 24
train_epoch = 500

# Hàm xây dựng model BiLSTM với các tham số từ PSO
def build_model(num_units_1, num_units_2, num_units_3, dropout_rate, learning_rate):
    model = Sequential([
        Bidirectional(LSTM(int(num_units_1), return_sequences=True, input_shape=(time_step, 1))),
        Dropout(dropout_rate),
        BatchNormalization(),

        Bidirectional(LSTM(int(num_units_2), return_sequences=True)),
        Dropout(dropout_rate),

        Bidirectional(LSTM(int(num_units_3))),
        Dropout(dropout_rate),

        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# Hàm đánh giá model (Hàm mục tiêu của PSO)
def evaluate_model(params):
    num_units_1, num_units_2, num_units_3, dropout_rate, learning_rate, batch_size = params
    model = build_model(num_units_1, num_units_2, num_units_3, dropout_rate, learning_rate)

    history = model.fit(
        x_train, y_train,
        epochs=20,  # Chạy nhanh để đánh giá
        batch_size=int(batch_size),
        validation_data=(x_test, y_test),
        verbose=0
    )
    return min(history.history['val_loss'])  # PSO sẽ tối ưu hóa giá trị nhỏ nhất của val_loss

def train():
    global x_train, y_train, x_test, y_test  # Cần dùng toàn cục cho PSO

    # Đọc dữ liệu
    train_data, test_data, scaler = read_data(file_path)
    x_train, y_train = create_dataset(train_data['scaled_count'].values, time_step)
    x_test, y_test = create_dataset(test_data['scaled_count'].values, time_step)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Tối ưu hyperparameters với PSO
    lb = [32, 16, 8, 0.1, 1e-5, 16]  # Giới hạn dưới
    ub = [256, 128, 64, 0.5, 1e-2, 128]  # Giới hạn trên

    best_params, _ = pso(evaluate_model, lb, ub, swarmsize=10, maxiter=5)
    num_units_1, num_units_2, num_units_3, dropout_rate, learning_rate, batch_size = best_params
    print(f"✅ Best Hyperparameters found: {best_params}")

    # Huấn luyện mô hình với best_params
    best_metrics = load_best_metrics(metrics_path)
    save_callback = CustomSaveCallback(x_test, y_test, scaler, best_metrics['r2_score'] if best_metrics else -np.inf,
                                       model_path, metrics_path)

    model = build_model(num_units_1, num_units_2, num_units_3, dropout_rate, learning_rate)
    model.fit(
        x_train, y_train,
        epochs=train_epoch,
        batch_size=int(batch_size),
        validation_data=(x_test, y_test),
        verbose=1,
        callbacks=save_callback
    )

    return test_data, y_test, scaler, save_callback.best_predictions

if __name__ == '__main__':
    test_data, test_targets, scaler, best_predictions = train()
    plot_results(test_data, test_targets, best_predictions, scaler, plot_path, metrics_path)

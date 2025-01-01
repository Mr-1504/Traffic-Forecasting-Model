import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('../resource/train27303.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.sort_values('timestamp', inplace=True)
data = data[(data['timestamp'] <= '2015-11-25')]
print(data)
plt.figure(figsize=(20, 10))
plt.plot(data['timestamp'], data['hourly_traffic_count'], label='Traffic Count', color='green')
plt.ylabel('Traffic Count')
plt.title('Hourly traffic count overtime')
plt.legend()
plt.savefig('hourly traffic count overtime.png')
plt.close()

train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]
plt.figure(figsize=(20, 10))
plt.plot(test_data['timestamp'], test_data['hourly_traffic_count'], label='Traffic Count', color='green')
plt.ylabel('Traffic Count')
plt.title('Hourly traffic count in test')
plt.legend()
plt.savefig('hourly traffic count in test.png')
plt.close()

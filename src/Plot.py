import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('../resource/train27303.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.sort_values('timestamp', inplace=True)
train_data = data[(data['timestamp'] <= '2015-11-25')]
print(data)
plt.figure(figsize=(20, 10))
plt.plot(train_data['timestamp'], train_data['hourly_traffic_count'], label='Traffic Count', color='green')
plt.ylabel('Traffic Count')
plt.title('Hourly traffic count in training')
plt.legend()
plt.savefig('hourly traffic count in training.png')
plt.close()



test_data = data[((data['timestamp'] > '2015-12-22') & (data['timestamp'] <= '2015-12-31'))]
plt.figure(figsize=(20, 10))
plt.plot(test_data['timestamp'], test_data['hourly_traffic_count'], label='Traffic Count', color='green')
plt.ylabel('Traffic Count', fontsize=18)  # 📌 Tăng cỡ chữ trục Y
plt.title('Hourly traffic count in testing', fontsize=22)
plt.legend()
plt.legend(fontsize=16)  # 📌 Tăng cỡ chữ trong legend
plt.xticks(fontsize=14)  # 📌 Tăng cỡ chữ cho trục X
plt.yticks(fontsize=14)
plt.savefig('../res/hourly traffic count in testing.png')
plt.close()

# def save_partial_table_as_png(df, filename="output.png", head=5, tail=5):
#     fig, ax = plt.subplots(figsize=(12, 6))  # 🟢 Tăng chiều cao
#     ax.axis("off")
#
#     # 🟢 Lấy 5 dòng đầu & 5 dòng cuối
#     partial_df = pd.concat([df.head(head), df.tail(tail)])
#
#     # 🟢 Chèn dấu "..." vào giữa
#     dots = pd.DataFrame([["...", "..."]], columns=df.columns)
#     final_df = pd.concat([partial_df.iloc[:head], dots, partial_df.iloc[head:]])
#
#     # 🟢 Màu nền xen kẽ cho dễ đọc
#     colors = [["#f5f5f5" if i % 2 == 0 else "#e0e0e0" for _ in range(final_df.shape[1])] for i in range(final_df.shape[0])]
#
#     # 🟢 Tạo bảng
#     table = ax.table(cellText=final_df.values,
#                      colLabels=final_df.columns,
#                      cellColours=colors,
#                      cellLoc="center",
#                      loc="center")
#
#     # 🟢 Tùy chỉnh bảng
#     table.auto_set_font_size(False)
#     table.set_fontsize(12)
#     table.auto_set_column_width(col=list(range(len(final_df.columns))))
#     table.scale(1.2, 2)  # 📌 Tăng khoảng cách giữa hàng
#
#     # 🟢 Định dạng header
#     for (i, key) in enumerate(final_df.columns):
#         cell = table[0, i]
#         cell.set_facecolor("#4CAF50")
#         cell.set_text_props(weight="bold", color="white")
#
#     # 🟢 Lưu file
#     plt.savefig(filename, dpi=300, bbox_inches="tight")
#     plt.show()
#
# save_partial_table_as_png(data, "output.png")

err_data = data[((data['timestamp'] > '2015-11-25') & (data['timestamp'] <= '2015-12-22'))]
print(err_data)
plt.figure(figsize=(20, 10))
plt.plot(err_data['timestamp'], err_data['hourly_traffic_count'], label='Traffic Count', color='green')
plt.ylabel('Traffic Count', fontsize=18)  # 📌 Tăng cỡ chữ trục Y
plt.title('Hourly traffic count in err', fontsize=22)
plt.legend()
plt.legend(fontsize=16)  # 📌 Tăng cỡ chữ trong legend
plt.xticks(fontsize=14)  # 📌 Tăng cỡ chữ cho trục X
plt.yticks(fontsize=14)
plt.savefig('../res/hourly traffic count in err.png')
plt.close()
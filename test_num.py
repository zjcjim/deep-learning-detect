import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import torch

def predict_num(num):
    # 数据准备
    tensor_num = torch.tensor(num, device='cuda')
    data = tensor_num.cpu().numpy()
    X = np.arange(len(data)).reshape(-1, 1)
    y = data

    # 构建线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    # 使用模型进行预测
    next_index = np.array([[len(data)]]) # 预测下一个数据点的索引
    prediction = model.predict(next_index)

    return prediction[0]

if __name__ == '__main__':
    # 运行预测函数，传入数据 [1, 2, 3, 4, 5, 6, 7, 8, 9]
    first_prediction = predict_num([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(f"第一个预测值: {first_prediction}")

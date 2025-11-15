import numpy as np

# Hồi quy tuyến tính bằng Gradient Descent
X = np.array([30, 32.4138, 34.8276, 37.2414, 39.6552, 42.069, 44.4828,
              46.8966, 49.3103, 51.7241])
y = np.array([448.524, 509.248, 535.104, 551.432, 623.418, 625.992,
              655.248, 701.377, 748.918, 757.881])

m = len(X)
X_b = np.c_[np.ones(m), X]

theta = np.random.randn(2)
learning_rate = 0.0001
epochs = 20000

for i in range(epochs):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_rate * gradients

w0, w1 = theta
print("w0 =", w0)
print("w1 =", w1)

# dự đoán
x_new = 50
y_pred = w0 + w1 * x_new
print("Giá dự đoán (Gradient Descent):", y_pred)


# Hồi quy tuyến tính bằng Sklearn
# from sklearn.linear_model import LinearRegression
#
# X = np.array([30, 32.4138, 34.8276, 37.2414, 39.6552, 42.069, 44.4828,
#               46.8966, 49.3103, 51.7241]).reshape(-1,1)
#
# y = np.array([448.524, 509.248, 535.104, 551.432, 623.418, 625.992,
#               655.248, 701.377, 748.918, 757.881])
#
# model = LinearRegression()
# model.fit(X, y)
#
# print("w0 =", model.intercept_)
# print("w1 =", model.coef_[0])
#
# y_pred = model.predict([[50]])
# print("Giá dự đoán (Sklearn):", y_pred[0])

#
# import numpy as np
#
# # ================================
# # 1. HÀM SIGMOID
# # ================================
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))
#
# # ================================
# # 2. LOSS (binary cross entropy)
# # ================================
# def loss(X, y, w, b):
#     z = X.dot(w) + b
#     p = sigmoid(z)
#     return -np.mean(y*np.log(p + 1e-9) + (1-y)*np.log(1-p + 1e-9))
#
# # ================================
# # 3. TRAIN BẰNG GRADIENT DESCENT
# # ================================
# def train_logistic(X, y, lr=0.1, epochs=5000):
#     n_features = X.shape[1]
#     w = np.zeros(n_features)
#     b = 0
#
#     for e in range(epochs):
#         z = X.dot(w) + b
#         p = sigmoid(z)
#
#         dw = np.dot(X.T, (p - y)) / len(y)
#         db = np.sum(p - y) / len(y)
#
#         w -= lr * dw
#         b -= lr * db
#
#         if e % 500 == 0:
#             print("Epoch", e, "Loss =", loss(X, y, w, b))
#
#     return w, b
#
# # ================================
# # 4. PREDICT
# # ================================
# def predict(X, w, b):
#     p = sigmoid(X.dot(w) + b)
#     return (p >= 0.5).astype(int), p
#
# # ================================
# # 5. DỮ LIỆU TỪ ĐỀ
# # ================================
# X = np.array([
#     [10, 1],
#     [5, 2],
#     [6, 1.8],
#     [7, 1],
#     [8, 2],
#     [9, 0.5],
#     [4, 3],
#     [5, 2.5],
#     [8, 1],
#     [4, 2.5],
#     [8, 0.1],
#     [7, 0.15],
#     [4, 1],
#     [5, 0.8],
#     [7, 0.3],
#     [4, 1],
#     [5, 0.5],
#     [6, 0.3],
#     [7, 0.2],
#     [8, 0.15]
# ], float)
#
# y = np.array([
#     1,1,1,1,1,1,1,1,1,1,
#     0,0,0,0,0,0,0,0,0,0
# ], float)
#
# # ================================
# # 6. HUẤN LUYỆN
# # ================================
# print("==== TRAINING LOGISTIC REGRESSION ====")
# w, b = train_logistic(X, y, lr=0.1, epochs=5000)
#
# print("\nTrọng số w =", w)
# print("Bias b =", b)
#
# # ================================
# # 7. DỰ ĐOÁN THỬ 1 MẪU
# # ================================
# test = np.array([6, 1.5])   # Lương = 6, thời gian = 1.5 năm
#
# pred, prob = predict(test, w, b)
#
# print("\nMẫu test:", test)
# print("Xác suất được cho vay =", prob)
# print("Kết luận (1 = cho vay, 0 = không) →", pred)

import numpy as np

# =======================================
# 1. Perceptron nhị phân
# =======================================
def train_binary_perceptron(X, y, epochs=200):
    w = np.zeros(X.shape[1])
    b = 0
    for _ in range(epochs):
        changed = False
        for xi, yi in zip(X, y):
            a = 1 if np.dot(w, xi) + b >= 0 else 0
            if a != yi:
                e = yi - a
                w += e * xi
                b += e
                changed = True
        if not changed:
            break
    return w, b

# =======================================
# 2. One-vs-Rest Perceptron đa lớp
# =======================================
def train_multiclass(X, y):
    classes = np.unique(y)
    models = {}

    for c in classes:
        y_binary = (y == c).astype(int)
        w, b = train_binary_perceptron(X, y_binary)
        models[c] = (w, b)

    return models

# =======================================
# 3. Dự đoán nhãn mới
# =======================================
def predict_multiclass(models, x):
    scores = {}
    for c, (w, b) in models.items():
        scores[c] = np.dot(w, x) + b
    return max(scores, key=scores.get)

# =======================================
# 4. DỮ LIỆU — CHỈ CẦN THAY PHẦN NÀY
# =======================================

# X = [Cao, Rộng, Chín]
X = np.array([
    [13, 12, 9],    # A
    [11, 13, 8],    # A
    [8, 8, 8],      # B
    [8, 9, 9],      # B
    [6, 7, 8],      # C
    [6, 7, 9],      # C
    [7, 6, 4],      # D
    [7, 7, 5],      # D
    [3, 1, 2],      # E
    [3, 2, 3],      # E
], float)

y = np.array(["A","A","B","B","C","C","D","D","E","E"])

# Mẫu cần dự đoán
test = np.array([3, 3, 5], float)

# =======================================
# 5. HUẤN LUYỆN & DỰ ĐOÁN
# =======================================
models = train_multiclass(X, y)
predict_label = predict_multiclass(models, test)

print("\nNhãn dự đoán cho mẫu", test, "là →", predict_label)

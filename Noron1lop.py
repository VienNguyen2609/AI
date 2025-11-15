# perceptron_full_vi.py
# Mã Perceptron tổng quát cho bài toán phân lớp nhị phân (0/1)
# Cách chạy: pip install numpy
#           python perceptron_full_vi.py

import numpy as np

def load_csv(path, delimiter=',', skip_header=1):
    """
    Đọc dữ liệu từ file CSV.
    File CSV: feat1,feat2,...,label   (label ở cột cuối, giá trị 0 hoặc 1)
    Trả về X (n_samples, n_features), y (n_samples,)
    """
    data = np.loadtxt(path, delimiter=delimiter, skiprows=skip_header)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y

def train_perceptron(X, y, max_epochs=1000, verbose=True):
    """
    Huấn luyện Perceptron theo qui tắc cập nhật cơ bản.
    X: numpy array (n_samples, n_features)
    y: numpy array (n_samples,) chứa 0 hoặc 1
    Trả về w (vector trọng số), b (bias), epochs_used, converged (bool)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)
    n_samples, n_features = X.shape

    # Khởi tạo trọng số = 0 và bias = 0 (cũng có thể ngẫu nhiên nhỏ)
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    for epoch in range(1, max_epochs + 1):
        changed = False
        for xi, yi in zip(X, y):
            n = np.dot(w, xi) + b
            a = 1 if n >= 0 else 0        # hàm step
            if a != yi:
                e = yi - a
                w = w + e * xi           # cập nhật trọng số
                b = b + e                # cập nhật bias
                changed = True
        if verbose:
            print(f"Epoch {epoch:4d}  cập nhật trong epoch: {changed}")
        if not changed:
            # Hội tụ: không có cập nhật trong một epoch
            if verbose:
                print(f"Hội tụ sau {epoch} epoch.")
            return w, b, epoch, True

    # Nếu chạy hết max_epochs mà chưa hội tụ
    if verbose:
        print(f"Không hội tụ sau {max_epochs} epoch.")
    return w, b, max_epochs, False

def predict_perceptron(X, w, b):

    X = np.atleast_2d(X)
    scores = X.dot(w) + b
    return (scores >= 0).astype(int), scores

if __name__ == "__main__":
    X_manual = np.array([[1,1],[8,3],[2,7],[8,8],[9,9]], dtype=float)
    y_manual = np.array([0,0,1,1,1], dtype=int)

    print("Bắt đầu huấn luyện Perceptron với dữ liệu thủ công...")
    w, b, epochs_used, converged = train_perceptron(X_manual, y_manual, max_epochs=1000, verbose=True)
    print("\nTrọng số học được:", w)
    print("Bias học được:", b)
    print("Epoch đã dùng:", epochs_used, "Hội tụ:", converged)

    # Dự đoán cho điểm test (mặc định theo đề: x1=9, x2=2)
    test_point = np.array([9.0, 2.0])
    pred, score = predict_perceptron(test_point, w, b)
    label_str = "Chín" if pred[0] == 1 else "Xanh"
    print(f"\nDự đoán cho điểm {test_point.tolist()} -> score={score[0]:.4f}, nhãn={pred[0]} ({label_str})")

    # In kết quả trên toàn tập huấn luyện để kiểm tra
    preds_train, scores_train = predict_perceptron(X_manual, w, b)
    print("\nDự đoán trên tập huấn luyện:")
    for xi, yi, pi, sc in zip(X_manual, y_manual, preds_train, scores_train):
        print(f" x={xi.tolist()}  y_true={yi}  -> score={sc:.4f}  pred={pi}")

    #--- Cách B: đọc dữ liệu từ CSV (bỏ comment nếu muốn dùng) ---
    #Ví dụ file CSV có header, cột cuối là label 0/1
    #path_csv = r"E:\Python\AI\Book1.csv"
    # X_csv, y_csv = load_csv(path_csv)
    # w2, b2, e2, conv2 = train_perceptron(X_csv, y_csv, max_epochs=2000, verbose=True)
    # print("Trọng số từ CSV:", w2, "bias:", b2)

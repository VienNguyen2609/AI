# perceptron_2neuron_vi.py
# Perceptron nhiều đầu ra (2 neuron) cho bài lớp 2-bit
# Chạy: pip install numpy
#       python perceptron_2neuron_vi.py

import numpy as np

def load_csv_multi(path, delimiter=',', skip_header=1, n_label_cols=2):
    """
    Đọc CSV: các cột cuối là nhãn (n_label_cols cột, mỗi cột 0/1).
    Trả về X (n_samples, n_features) và Y (n_samples, n_label_cols)
    """
    data = np.loadtxt(path, delimiter=delimiter, skiprows=skip_header)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    X = data[:, :-n_label_cols]
    Y = data[:, -n_label_cols:].astype(int)
    return X, Y

def train_perceptron_multi(X, Y, max_epochs=1000, verbose=True):
    """
    Huấn luyện perceptron cho nhiều neuron đầu ra cùng lúc.
    X: (n_samples, n_features)
    Y: (n_samples, n_outputs) với 0/1
    Trả về W (n_features, n_outputs), b (n_outputs,), epochs_used, converged
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=int)
    n, d = X.shape
    _, m = Y.shape

    # Khởi tạo trọng số cho mỗi neuron (cột trên W) và bias
    W = np.zeros((d, m), dtype=float)
    b = np.zeros(m, dtype=float)

    for epoch in range(1, max_epochs + 1):
        changed = False
        for xi, yi in zip(X, Y):
            scores = xi.dot(W) + b        # (m,)
            preds = (scores >= 0).astype(int)
            # cập nhật từng neuron nếu sai
            dif = yi - preds              # vector (m,)
            if np.any(dif != 0):
                # cập nhật vector: W[:,j] += dif[j] * xi
                W += np.outer(xi, dif)
                b += dif
                changed = True
        if verbose:
            print(f"Epoch {epoch:4d}  cập nhật trong epoch: {changed}")
        if not changed:
            if verbose:
                print(f"Hội tụ sau {epoch} epoch.")
            return W, b, epoch, True

    if verbose:
        print(f"Không hội tụ sau {max_epochs} epoch.")
    return W, b, max_epochs, False

def predict_multi(X, W, b):
    X = np.atleast_2d(X)
    scores = X.dot(W) + b
    preds = (scores >= 0).astype(int)
    return preds, scores

if __name__ == "__main__":
    # ---------------- Cách A: nhập thủ công (thay X_manual, Y_manual nếu cần) ----------------
    # Dữ liệu tương tự bảng bạn gửi (2 feature: cân nặng, độ chín; 2-bit label)
    X_manual = np.array([
        [1,1],
        [2,2],
        [8,1],
        [9,3],
        [2,7],
        [3,8],
        [8,8],
        [9,9]
    ], dtype=float)
    # Nhãn theo thứ tự: cột1 cột2  (ví dụ 00,00,10,...)
    Y_manual = np.array([
        [0,0],
        [0,0],
        [1,0],
        [1,0],
        [0,1],
        [0,1],
        [1,1],
        [1,1]
    ], dtype=int)

    W, b, e, conv = train_perceptron_multi(X_manual, Y_manual, max_epochs=1000, verbose=True)
    print("\nTrọng số W:\n", W)
    print("Bias b:", b)

    test = np.array([9.0, 2.0])
    pred_test, score_test = predict_multi(test, W, b)
    print(f"\nTest point {test.tolist()} -> scores={score_test.tolist()}, preds={pred_test.tolist()}")
    print("Giải thích nhãn 2-bit: [bit1, bit2] -> tương ứng lớp (ví dụ 10 = bit1=1,bit2=0)")

    preds_train, scores_train = predict_multi(X_manual, W, b)
    print("\nDự đoán trên tập huấn luyện:")
    for xi, yi, pi, sc in zip(X_manual, Y_manual, preds_train, scores_train):
        print(f" x={xi.tolist()} y_true={yi.tolist()} -> scores={sc.tolist()} pred={pi.tolist()}")

    # ---------------- Cách B: đọc từ CSV ----------------
    # Nếu muốn dùng CSV, chuẩn bị file dạng:
    # feat1,feat2,...,label1,label2
    # Ví dụ header: diameter,color,label1,label2
    # Mỗi dòng: 4,2,0,0
    # Thay đường dẫn phù hợp cho máy bạn, lưu ý escape backslash hoặc dùng raw string r"..."
    # path_csv = r"E:\Python\AI\Book1_2bit.csv"   # <--- sửa đường dẫn file CSV của bạn ở đây
    # try:
    #     X_csv, Y_csv = load_csv_multi(path_csv, delimiter=',', skip_header=1, n_label_cols=2)
    #     print("\nĐọc CSV thành công, kích thước:", X_csv.shape, Y_csv.shape)
    #     W2, b2, e2, conv2 = train_perceptron_multi(X_csv, Y_csv, max_epochs=2000, verbose=True)
    #     print("\nTrọng số từ CSV:\n", W2)
    #     print("Bias từ CSV:", b2)
    # except Exception as ex:
    #     print("\nKhông thể đọc/huấn luyện từ CSV:", ex)
    #     print("Kiểm tra: file tồn tại, định dạng có header và 2 cột label cuối là 0/1.")

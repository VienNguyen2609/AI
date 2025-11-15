from math import log2
from collections import Counter
import json

# ============================
# 1. HÀM ENTROPY
# ============================
def entropy(y):
    counts = Counter(y)
    total = len(y)
    return -sum((count/total) * log2(count/total) for count in counts.values())

# ============================
# 2. HÀM TÍNH GAIN
# ============================
def information_gain(X, y, attr):
    total_entropy = entropy(y)
    values = set(x[attr] for x in X)
    weighted_entropy = 0

    for v in values:
        subset_y = [y[i] for i in range(len(y)) if X[i][attr] == v]
        weighted_entropy += (len(subset_y)/len(y)) * entropy(subset_y)

    return total_entropy - weighted_entropy


# ============================
# 3. THUẬT TOÁN ID3
# ============================
def id3(X, y, attributes):
    # Nếu chỉ còn 1 lớp → là nút lá
    if len(set(y)) == 1:
        return y[0]

    # Hết thuộc tính → chọn lớp xuất hiện nhiều nhất
    if not attributes:
        return Counter(y).most_common(1)[0][0]

    # Tính gain cho từng thuộc tính
    gains = {attr: information_gain(X, y, attr) for attr in attributes}
    best_attr = max(gains, key=gains.get)

    tree = {best_attr: {}}

    # Tạo nhánh cho mỗi giá trị của thuộc tính chọn
    for value in set(x[best_attr] for x in X):
        subset_X = [x for x in X if x[best_attr] == value]
        subset_y = [y[i] for i, x in enumerate(X) if x[best_attr] == value]

        remaining_attrs = [a for a in attributes if a != best_attr]

        tree[best_attr][value] = id3(subset_X, subset_y, remaining_attrs)

    return tree


# ============================
# 4. HÀM DỰ ĐOÁN
# ============================
def predict(tree, sample):
    # Nút lá
    if type(tree) != dict:
        return tree

    attr = next(iter(tree))               # thuộc tính của nút hiện tại
    value = sample.get(attr, None)        # giá trị thuộc tính của mẫu

    if value not in tree[attr]:
        return "Không có nhánh phù hợp"

    return predict(tree[attr][value], sample)


# ============================
# 5. DỮ LIỆU THEO ĐỀ BÀI
# ============================
X = [
    {"Màu":"Xanh", "Cao":"Cao", "Hoạt động":"Nhiều", "Môn":"Cầu lông"},
    {"Màu":"Vàng", "Cao":"Thấp", "Hoạt động":"Ít", "Môn":"Chạy bộ"},
    {"Màu":"Xanh", "Cao":"Thấp", "Hoạt động":"TB", "Môn":"Chạy bộ"},
    {"Màu":"Trắng", "Cao":"Thấp", "Hoạt động":"Ít", "Môn":"Cầu lông"},
    {"Màu":"Vàng", "Cao":"Cao", "Hoạt động":"Ít", "Môn":"Cầu lông"},
    {"Màu":"Vàng", "Cao":"Thấp", "Hoạt động":"TB", "Môn":"Cầu lông"},
    {"Màu":"Trắng", "Cao":"Cao", "Hoạt động":"Nhiều", "Môn":"Chạy bộ"},
    {"Màu":"Vàng", "Cao":"TB", "Hoạt động":"TB", "Môn":"Chạy bộ"},
    {"Màu":"Xanh", "Cao":"TB", "Hoạt động":"Nhiều", "Môn":"Cầu lông"}
]

y = ["1","1","1","2","2","2","2","2","1"]


# ============================
# 6. HUẤN LUYỆN CÂY QUYẾT ĐỊNH
# ============================
attributes = list(X[0].keys())
tree = id3(X, y, attributes)

print("===== CÂY QUYẾT ĐỊNH (ID3) =====")
print(json.dumps(tree, indent=2, ensure_ascii=False))


# ============================
# 7. DỰ ĐOÁN MẪU THỨ 10
# ============================
test_sample = {
    "Màu": "Trắng",
    "Cao": "Cao",
    "Hoạt động": "Nhiều",
    "Môn": "Chạy bộ"
}

print("\n===== DỰ ĐOÁN MẪU THỨ 10 =====")
result = predict(tree, test_sample)
print("Loại dự đoán:", result)

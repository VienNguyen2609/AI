from math import log2
from collections import Counter
import json

def entropy(y):
    counts = Counter(y)
    total = len(y)
    return -sum((count/total) * log2(count/total) for count in counts.values())

def information_gain(X, y, attr):
    total_entropy = entropy(y)
    values = set(x[attr] for x in X)
    weighted_entropy = 0
    for v in values:
        subset_y = [y[i] for i in range(len(y)) if X[i][attr] == v]
        weighted_entropy += (len(subset_y)/len(y)) * entropy(subset_y)
    return total_entropy - weighted_entropy


def id3(X, y, attributes):
    if len(set(y)) == 1:
        return y[0]
    if not attributes:
        return Counter(y).most_common(1)[0][0]

    gains = {attr: information_gain(X, y, attr) for attr in attributes}
    best_attr = max(gains, key=gains.get)

    # Tạo node cho attr tốt nhất
    tree = {best_attr: {}}
    for value in set(x[best_attr] for x in X):
        subset_X = [x for x in X if x[best_attr] == value]
        subset_y = [y[i] for i, x in enumerate(X) if x[best_attr] == value]
        remaining_attrs = [a for a in attributes if a != best_attr]
        tree[best_attr][value] = id3(subset_X, subset_y, remaining_attrs)
    return tree

X = [
    {"Outlook": "Sunny", "Temperature": 85, "Humidity": 85, "Wind": "Weak"},
    {"Outlook": "Sunny", "Temperature": 80, "Humidity": 90, "Wind": "Strong"},
    {"Outlook": "Overcast", "Temperature": 83, "Humidity": 78, "Wind": "Weak"},
    {"Outlook": "Rain", "Temperature": 70, "Humidity": 96, "Wind": "Weak"},
    {"Outlook": "Rain", "Temperature": 68, "Humidity": 80, "Wind": "Weak"},
    {"Outlook": "Rain", "Temperature": 65, "Humidity": 70, "Wind": "Strong"},
    {"Outlook": "Overcast", "Temperature": 64, "Humidity": 65, "Wind": "Strong"},
    {"Outlook": "Sunny", "Temperature": 72, "Humidity": 95, "Wind": "Weak"},
    {"Outlook": "Sunny", "Temperature": 69, "Humidity": 70, "Wind": "Weak"},
    {"Outlook": "Rain", "Temperature": 75, "Humidity": 80, "Wind": "Weak"},
    {"Outlook": "Sunny", "Temperature": 75, "Humidity": 70, "Wind": "Strong"},
    {"Outlook": "Overcast", "Temperature": 72, "Humidity": 90, "Wind": "Strong"},
    {"Outlook": "Overcast", "Temperature": 81, "Humidity": 75, "Wind": "Weak"},
    {"Outlook": "Rain", "Temperature": 71, "Humidity": 91, "Wind": "Strong"}
]
y = ["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No"]
# -----------------------
# 5. HUẤN LUYỆN & IN CÂY
# -----------------------
attributes = list(X[0].keys())
tree = id3(X, y, attributes)
print(json.dumps(tree, indent=2, ensure_ascii=False))

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

Dataframe = pd.read_csv(rf"heart_disease_risk_dataset_earlymed.csv")
# print(Dataframe.head())

X_df = Dataframe.drop("Heart_Risk", axis=1)
# print(f"X_DataFrame Column: \n{X_df.columns}")

X = X_df
y = Dataframe["Heart_Risk"]

info_gain = mutual_info_classif(X=X, y=y)
# print(f"InformationGain: \n{info_gain}")

info_gain_pairs = zip(X.columns, info_gain)
sorted_info_gain = sorted(info_gain_pairs, key=lambda x: x[1], reverse=True)

print("\nInformation Gain sorted from highest to lowest:")
result = {"Features": [], "Information Gain": []}

for feature, ig in sorted_info_gain:
    result["Features"].append(feature)
    result["Information Gain"].append(ig)


df_result = pd.DataFrame(result)
print(f"Dataframe: \n{df_result}")

Feature_IG = [
    "Age",
    "Chest_Pain",
    "Cold_Sweats_Nausea",
    "Dizziness",
    "Shortness_of_Breath",
    "Fatigue",
    "Pain_Arms_Jaw_Back",
    "Palpitations",
    "Swelling",
]
model = RandomForestClassifier(random_state=42)
X = Dataframe[Feature_IG].values
y = Dataframe["Heart_Risk"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")

# Define class labels (เปลี่ยนตรงนี้ให้เป็นคลาสของคุณ)
class_labels = ["High Heart Risk", "Low Heart Risk"]  # แก้ไขให้ตรงกับข้อมูลของคุณ

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(7, 5))
plt.imshow(conf_matrix, cmap="Pastel1")
plt.colorbar()

# ใส่ตัวเลขลงในช่องของ confusion matrix
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(x=j, y=i, s=conf_matrix[i, j], ha="center", va="center", fontsize=12)

# ตั้งค่า Label ให้แกน x และ y
plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels)
plt.yticks(ticks=np.arange(len(class_labels)), labels=class_labels)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()

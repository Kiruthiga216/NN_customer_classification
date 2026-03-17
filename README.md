# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="737" height="993" alt="image" src="https://github.com/user-attachments/assets/a01c28f9-55ed-400e-8740-ec417fca47e1" />



## DESIGN STEPS

### STEP 1:  

Import the required libraries for data handling and neural networks

### STEP 2:

Load the dataset and explore its structure.

### STEP 3:

Clean the dataset and handle missing values if present.

## STEP 4:

Encode categorical variables into numerical format.

## STEP 5:

Normalize or scale the numerical features

## STEP 6:

Split the dataset into training and testing sets.

## STEP 7:

Define the neural network architecture (64 → 32 → 16 → 8 → 4).

## STEP 8:

Select CrossEntropyLoss as the loss function and Adam as the optimizer.

## STEP 9:

Train the model using forward pass, loss calculation, backpropagation, and weight updates.

## STEP 10:

Evaluate the model using accuracy, confusion matrix, and classification report.



### Name:  KIRUTHIGA.B
### Register Number: 212224040160

## PROGRAM

```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('/content/customers.csv')
print(data.head(10))

data.fillna({
    "Work_Experience": 0,
    "Family_Size": data["Family_Size"].median()
}, inplace=True)

categorical_columns = [
    "Gender", "Ever_Married", "Graduated",
    "Profession", "Spending_Score", "Var_1"
]

for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

label_encoder = LabelEncoder()
data["Segmentation"] = label_encoder.fit_transform(data["Segmentation"])


X = data.drop(columns=["Segmentation"])
y = data["Segmentation"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)


class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")


model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, epochs=50)


model.eval()
predictions, actuals = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())

accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(
    actuals,
    predictions,
    target_names=label_encoder.classes_
)

print("Name: Kiruthiga.B")
print("Register No: 212224040160")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


sns.heatmap(
    conf_matrix,
    annot=True,
    cmap='Blues',
    fmt='g',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


d_class_index = label_encoder.transform(['D'])[0]


d_sample_index = (y_test == d_class_index).nonzero(as_tuple=True)[0][0].item()

sample_input = X_test[d_sample_index].unsqueeze(0)

with torch.no_grad():
    output = model(sample_input)
    predicted_class_index = torch.argmax(output, dim=1).item()

predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
actual_label = label_encoder.inverse_transform([y_test[d_sample_index].item()])[0]

print("Name: Kiruthiga.B")
print("Register No: 212224040160")
print(f"Predicted class: {predicted_label}")
print(f"Actual class: {actual_label}")
```

## Dataset Information

<img width="918" height="697" alt="image" src="https://github.com/user-attachments/assets/6ecdbb47-b8ba-4959-a3e9-d17d0ccdf4b0" />


## OUTPUT

<img width="904" height="682" alt="image" src="https://github.com/user-attachments/assets/91bea522-239d-4e0a-b61c-4a450f2da02e" />


### Confusion Matrix

<img width="719" height="623" alt="image" src="https://github.com/user-attachments/assets/d43411ec-07ab-485f-94fd-9c2566964057" />


### Classification Report

<img width="608" height="453" alt="image" src="https://github.com/user-attachments/assets/62dea76f-34b6-4bb3-a7c4-4501eabe833e" />



### New Sample Data Prediction

<img width="303" height="104" alt="image" src="https://github.com/user-attachments/assets/47a29e29-6a59-4369-9b2b-1bddba9b5324" />


## RESULT

Thus, a neural network classification model for the given dataset is To developed successfully.

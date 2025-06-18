# 🌸 Artificial Neural Network (ANN) Implementation – Iris Flower Classification

This project demonstrates how to build a **multi-layer Artificial Neural Network (ANN)** using **Keras (TensorFlow backend)** to classify **iris flowers** based on their physical features.

It uses the classic **Iris dataset** to train, test, and evaluate the model’s performance in recognizing flower species.

---

📌 Problem Statement

- Goal: Classify iris flowers into one of **three species**:
  - Setosa
  - Versicolor
  - Virginica

- Features used:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width

---

🔧 Methodology

- **Dataset Source**: `sklearn.datasets.load_iris()`
- **Samples**: 150
- **Features**: 4
- **Class Labels**: 3 (multi-class classification)

### 🔄 Data Preprocessing

- Data shuffled to ensure randomness
- Feature scaling using `StandardScaler`
- One-hot encoding of target classes using `LabelBinarizer`
- Train-Test split: `70% training / 30% testing`

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split

# Load and prepare the data
data = load_iris()
X = data.data
y = data.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode target
encoder = LabelBinarizer()
y_encoded = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
```

---

🧠 Neural Network Architecture

Built using the **Keras Sequential API**:

- **Input Layer**: 4 neurons (for 4 input features)
- **Hidden Layers**:
  - `Dense(12, activation='relu')`
  - `Dense(15, activation='relu')`
  - `Dense(8, activation='relu')`
  - `Dense(10, activation='relu')`
- **Output Layer**:
  - `Dense(3, activation='softmax')` for multi-class classification

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(12, input_shape=(4,), activation='relu'),
    Dense(15, activation='relu'),
    Dense(8, activation='relu'),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])
```

---

⚙️ Model Training

- **Loss Function**: `categorical_crossentropy`
- **Optimizer**: `adam`
- **Metric**: `accuracy`
- **Epochs**: `120`

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=120, verbose=1)
```

---

📊 Output & Evaluation

- Predicts flower species using:

```python
import numpy as np
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
```

- Final **accuracy** and **validation accuracy** are printed during training.
- You can visualize training history:

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

---

▶️ How to Run

1. **Install required libraries**:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

2. **Run the notebook** in:
   - Jupyter Notebook
   - Google Colab (recommended for ease)

3. **Upload or run the script** and observe learning metrics and prediction output.

---

🚀 Future Improvements

- Add early stopping and learning rate scheduling
- Try different activation functions or architectures
- Use `KFold` cross-validation for robust evaluation
- Deploy as a web app using Flask or Streamlit

---

🎯 Use Case

This project shows how deep learning can solve real-world classification problems using minimal code and open-source tools — a great starting point for any ML enthusiast.


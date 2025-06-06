# Artificial-Neural-Network-Implementation

This project demonstrates how to build a multi-layer Artificial Neural Network (ANN) using Keras to classify iris flowers based on their physical features. It uses the classic Iris dataset to train and evaluate the model.

📌 Problem Statement
Classify iris flowers into one of three species (Setosa, Versicolor, Virginica) using the following features:

Sepal length

Sepal width

Petal length

Petal width

🔧 Methodology
1. Dataset
Source: sklearn.datasets.load_iris()

150 samples, 4 features, 3 class labels.

2. Preprocessing
Data shuffled for randomness.

Feature scaling using StandardScaler.

Label encoding using LabelBinarizer to one-hot encode target classes.

3. Train-Test Split
70% training, 30% testing using train_test_split.

🧠 Neural Network Architecture
Built using Keras Sequential API:

Input Layer: 4 neurons (for 4 features)

Hidden Layers:

Dense(12), activation='relu'

Dense(15), activation='relu'

Dense(8), activation='relu'

Dense(10), activation='relu'

Output Layer:

Dense(3), activation='softmax' (for 3 classes)

⚙️ Model Training
Loss Function: categorical_crossentropy

Optimizer: adam

Metric: accuracy

Epochs: 120

Validation performed using test set.

✅ Output
Model predicts species of iris flower using model.predict_classes().

Final accuracy and validation scores track learning performance.

▶️ How to Run
Install required libraries:

pip install numpy pandas scikit-learn tensorflow
Run the code in Jupyter Notebook or Google Colab.

Optionally visualize training history (history.history) to see learning progress.

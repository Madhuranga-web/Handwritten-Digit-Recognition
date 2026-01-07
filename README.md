🖋️ Handwritten Digit Recognition using Machine Learning
This is a beginner-friendly Machine Learning project that classifies handwritten digits (0-9) using the Scikit-learn library. It uses a Support Vector Machine (SVM) classifier to achieve high accuracy.

🚀 Overview
The goal of this project is to train a model that can look at a small image of a handwritten digit and correctly identify which number it is. We use the classic MNIST digits dataset, which contains 1,797 8x8 pixel images.

🛠️ Tech Stack
Language: Python

Libraries: * Scikit-learn (Machine Learning Framework)

NumPy (Data Manipulation)

Matplotlib (Data Visualization)

Pillow (Image processing for custom input)

📊 How it Works
Data Loading: Loads the digits dataset from sklearn.datasets.

Preprocessing: Flattens the 8x8 image matrices into 1D arrays of 64 pixels.

Training: Splitting the data into 80% training and 20% testing sets. We use the Support Vector Classifier (SVC) to learn the patterns.

Evaluation: Calculating the accuracy score and displaying a Confusion Matrix to see where the model gets confused.

📈 Results
Accuracy: ~98% (Mention your actual accuracy here)

The model performs exceptionally well on standard digits but may vary with custom handwritten inputs due to the low resolution (8x8).

💻 How to Run
Clone this repository:

Bash

git clone https://github.com/your-username/Handwritten-Digit-Recognition.git
Install dependencies:

Bash

pip install numpy scikit-learn matplotlib pillow
Run the script:

Bash

python main.py
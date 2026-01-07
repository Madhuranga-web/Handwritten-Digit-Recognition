import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# 1.  (Load Data)
digits = datasets.load_digits()

# 2.  (Flattening)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# 3. (Train/Test Split)
# 70% , 30% checking without shuffling
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=False
)

# 4. Model  (SVC Algorithm from SVM)
clf = svm.SVC(gamma=0.001)

# Model (Training)
clf.fit(X_train, y_train)

# 5. (Prediction)
predicted = clf.predict(X_test)

# 6. image Visualization
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Predicted: {prediction}")

print("The task was successful! The numbers identified by the model are now visible.")
plt.show()
# Accuracy Score (How well the model performed)
accuracy = metrics.accuracy_score(y_test, predicted)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix Visualization
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
plt.show()

from PIL import Image
import numpy as np

def predict_my_digit(image_path):
    # 1. gray scale 
    img = Image.open(image_path).convert('L')
    # 2. Resize to 8x8
    img = img.resize((8, 8), Image.Resampling.LANCZOS)
    
    # 3. Convert to numpy array
    img_array = np.array(img)
    # Invert colors and scale to match training data
    img_array = 16 - (img_array / 16.0) 
    
   
    img_flat = img_array.reshape(1, -1)
    
    # 4. Model 
    prediction = clf.predict(img_flat)
    
    # image presentation
    plt.imshow(img_array, cmap='gray')
    plt.title(f"I think this is a: {prediction[0]}")
    plt.show()
    
    print(f"Model's prediction: {prediction[0]}")

# Usage:
# Provide the path to your image (e.g., 'digit.png')
# predict_my_digit('your_image_name.png')
🍔 Fast Food Image Classification using ResNet-50 (PyTorch)

This project implements a deep learning–based image classification system for fast food images using ResNet-50 with transfer learning and fine-tuning in PyTorch.
The model is trained on a custom fast food dataset and evaluated using accuracy, classification reports, and confusion matrices.

📌 Features

✅ Uses ResNet-50 pretrained on ImageNet

✅ Transfer Learning

Head (Fully Connected Layer) training

Full model fine-tuning

✅ Proper data preprocessing & normalization

✅ Performance evaluation using:

Accuracy

Classification Report

Confusion Matrix

✅ Visualization of:

Training & Validation Accuracy

Training & Validation Loss

Sample Predictions

✅ GPU support (CUDA if available)

🧠 Model Architecture

Backbone: ResNet-50 (ImageNet pretrained)

Input Size: 224 × 224 RGB images

Final Layer: Fully Connected Layer customized for number of food classes

⚙️ Requirements

Install required libraries using:

pip install torch torchvision matplotlib scikit-learn

🚀 Training Strategy
🔹 Step 1: Head Training

Freeze all ResNet-50 layers

Train only the final fully connected layer

Learning Rate: 1e-4

Epochs: 2

🔹 Step 2: Fine-Tuning

Unfreeze entire network

Train all layers with smaller learning rate

Learning Rate: 1e-5

Epochs: 2

📊 Evaluation Metrics

Training Accuracy

Validation Accuracy

Test Accuracy

Classification Report (Precision, Recall, F1-Score)

Confusion Matrix

📈 Results Visualization

Accuracy vs Epoch Curve

Loss vs Epoch Curve

Confusion Matrix

Sample Predictions from Test Set

⏱ Training Time

The script reports:

Head Training Time

Fine-Tuning Time

Total Training Time

🧪 Sample Predictions

Random test images are displayed with:

True Label

Predicted Label

🖥 Device Support

Automatically uses:

GPU (CUDA) if available

Otherwise CPU

📌 How to Run

Update dataset paths in the script:

train_path = "path/to/train"
val_path   = "path/to/valid"
test_path  = "path/to/test"


Run the Python script:

python resnet50_fast_food_classification.py

📚 Technologies Used

Python

PyTorch

Torchvision

Scikit-Learn

Matplotlib

🎓 Academic Context

This project was developed as part of an Artificial Neural Network (ANN) course and demonstrates:

CNN-based image classification

Transfer learning

Fine-tuning deep networks

Performance evaluation techniques

✨ Author

Minahil
BS Artificial Intelligence

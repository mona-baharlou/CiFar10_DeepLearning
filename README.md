# CIFAR-10 Image Classification Project with Deep Neural Network 

This project implements a deep learning model for classifying images from the CIFAR-10 dataset. The goal of this project is to understand and implement the fundamental steps in building and training a deep neural network from scratch, from data preprocessing to model evaluation.

## Features and What I Learned in This Project 

Throughout this project, I learned and implemented the following key skills and concepts:

1.  **Using Online Datasets (CIFAR-10)**

      * How to load and work with standard deep learning datasets available online, such as [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

2.  **Data Normalization**

      * The importance of normalizing image pixel values to improve the stability and speed of model training. This helps the neural network converge better.

3.  **One-Hot Encoding Conversion**

      * Converting categorical labels to One-Hot Encoding format, which is essential for categorical cross-entropy loss functions in deep neural networks.

4.  **Building a Sequential Model from Scratch**

      * Designing and constructing a deep neural network (Sequential Model) using various layers (e.g., Dense, Conv2D, MaxPooling, etc.) from the ground up, without using pre-built models.

5.  **Managing Overfitting with Dropout and Batch Normalization**

      * Identifying and diagnosing the phenomenon of **Overfitting** during training.
      * Implementing regularization techniques like **Dropout** to prevent overfitting and improve the model's generalization ability.
      * Using **Batch Normalization** to stabilize training and reduce the need for precise weight initialization.
  
     ![Overfitted](https://github.com/mona-baharlou/CiFar10_DeepLearning/blob/main/overfitting.PNG)
     ![After Batch Normalization & Dropout](https://github.com/mona-baharlou/CiFar10_DeepLearning/blob/main/ovirfitting_uopdate_pic.PNG)
    

7.  **Model Compilation**

      * Configuring and compiling the model by selecting an appropriate **Loss Function**, **Optimizer**, and **Metrics** for the classification problem.

8.  **Model Training**

      * The process of training the model on training and validation data, monitoring training progress at each **Epoch**.

9.  **Displaying and Evaluating Model Performance**

      * **Graphical Performance Display:** Plotting Loss and Accuracy graphs over epochs to visualize the training trend and detect overfitting.
      * **Confusion Matrix:** Generating and analyzing a **Confusion Matrix** to accurately assess the model's performance in classifying each category and identify systematic errors.
      * **Qualitative Evaluation:** Visually inspecting samples of images and the model's predictions for a more intuitive understanding of its performance.
  
    ![After Batch Normalization & Dropout](https://github.com/mona-baharlou/CiFar10_DeepLearning/blob/main/matrix.png)


    ![After Batch Normalization & Dropout](https://github.com/mona-baharlou/CiFar10_DeepLearning/blob/main/qual.png)

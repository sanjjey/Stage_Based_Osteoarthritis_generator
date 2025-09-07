# Conditional Generation of Osteoarthritis Progression using GANs

This project utilizes a Conditional Generative Adversarial Network (cGAN) to generate realistic knee X-ray images, conditioned on the specific stage of osteoarthritis (OA) and the side of the knee (left/right). The primary goal is to visualize the progression of the disease from a healthy state (Stage 0) to a severe state (Stage 4).

Additionally, the project includes an advanced application that can take a real-world X-ray of a healthy knee and generate a plausible future progression of the disease for that specific individual.

![Showcase of generated X-ray images](https://github.com/sanjjey/Stage_Based_Osteoarthritis_generator/blob/main/assets/progression_result.png?raw=true)
*An example of the model generating a 5-stage progression from a single input image.*

---

## 📜 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
  - [1. Training the Model](#1-training-the-model)
  - [2. Generating Progression from a Real Image](#2-generating-progression-from-a-real-image)
- [Technologies Used](#-technologies-used)

---

## 🚀 Project Overview

Osteoarthritis is a degenerative joint disease, and its progression is typically monitored through radiographic imaging. This project explores the use of deep learning, specifically cGANs, to model and visualize this progression. By learning the underlying data distribution of thousands of knee X-rays, the model can generate high-fidelity images for any given stage of the disease, providing a valuable tool for medical education, patient communication, and potentially for data augmentation.

---

## ✨ Key Features

- **Conditional Image Generation**: Creates knee X-rays for specific OA stages (0-4) and sides (left/right).
- **Disease Progression Visualization**: Generates a coherent sequence of images showing the disease's advancement.
- **Image-to-Image Translation**: An advanced application that takes a real healthy knee X-ray as input and generates its future OA progression by finding its representation in the GAN's latent space.
- **Deep Learning Architecture**: Implemented using PyTorch with a standard Deep Convolutional GAN (DCGAN) structure.

---

## 🏗️ Model Architecture

The project is built on a Conditional GAN framework, which consists of two main neural networks trained in an adversarial process:

### 1. Generator
A deep convolutional neural network that uses **transposed convolution** (upsampling) layers.
- **Input**: A 128-dimensional random noise vector (`z`) and two conditions (OA stage, knee side).
- **Function**: It learns to transform the input vector into a realistic 128x128 pixel grayscale image that matches the given conditions.

### 2. Discriminator
A standard **Convolutional Neural Network (CNN)**.
- **Input**: A 128x128 image (either real or generated) and its corresponding conditions.
- **Function**: It learns to classify whether a given image is a real X-ray from the dataset or a fake one created by the generator.

The training process involves these two networks competing: the Generator tries to fool the Discriminator, and the Discriminator tries to get better at catching the fakes.

---

## 💾 Dataset

This project uses the **Osteoarthritis Initiative (OAI) Dataset**, which is a large, publicly available collection of knee X-ray images. The dataset is sourced directly via the `kagglehub` library. The file names in the dataset contain the labels for the Kellgren-Lawrence (KL) grade (OA stage) and the knee side, which are essential for the conditional training.

---

## 🔧 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/sanjjey/Stage_Based_Osteoarthritis_generator.git](https://github.com/sanjjey/Stage_Based_Osteoarthritis_generator.git)
    cd Stage_Based_Osteoarthritis_generator
    ```

2.  **Install the required libraries:**
    It is recommended to use a virtual environment.
    ```bash
    pip install torch torchvision numpy matplotlib pillow tqdm kagglehub
    ```

---

## 🏃 Usage

There are two main parts to this project: training the model and using the trained model for progression generation.

### 1. Training the Model

-   Run the `OAI.ipynb` Jupyter notebook.
-   This notebook will automatically download the dataset from KaggleHub.
-   The training loop will run for the specified number of epochs (default is 500).
-   Checkpoints of the trained models will be saved in the `results/` directory.

### 2. Generating Progression from a Real Image

-   Use the second Jupyter notebook for the application (e.g., `Application.ipynb`).
-   **Important**: Update the `checkpoint_path` variable to point to your saved generator model (e.g., `./results/checkpoint_epoch_500.pth`).
-   Update the `input_image_path` to the location of a healthy (Stage 0) knee X-ray you want to use as input.
-   Run the cells. The code will perform an optimization process called **Latent Space Projection** to find the "seed" corresponding to your image and then generate the 5-stage progression.

---

## 🛠️ Technologies Used

-   **Python 3.x**
-   **PyTorch**: The primary deep learning framework.
-   **Jupyter Notebook**: For interactive development and experimentation.
-   **KaggleHub**: For easy and reproducible dataset access.
-   **Matplotlib**: For data visualization.
-   **NumPy**: For numerical operations.
-   **Pillow (PIL)**: For image manipulation.

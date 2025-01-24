# LeNet-5 in PyTorch

Welcome to the LeNet-5 implementation! This is a simple and beginner-friendly PyTorch implementation of the classic **LeNet-5** architecture, designed for the **MNIST handwritten digit classification** task. Whether you're new to deep learning or just want to understand how CNNs work, this is a great place to start.

---

## What is LeNet-5?

LeNet-5 is one of the earliest convolutional neural networks (CNNs), introduced by Yann LeCun in 1998. It was originally designed to recognize handwritten digits and is a great introduction to CNNs. The architecture is simple but powerful, making it perfect for learning.

---

## How Does It Work?

The LeNet-5 model consists of:
1. **Convolutional Layers**: Extract features from the input image.
2. **Pooling Layers**: Reduce the spatial dimensions of the feature maps.
3. **Fully Connected Layers**: Combine features to make predictions.

Here’s the breakdown of the architecture:
- Input: 32x32 grayscale image (MNIST images are resized from 28x28).
- Convolutional Layer 1: 6 filters of size 5x5, followed by ReLU activation.
- Pooling Layer 1: 2x2 average pooling.
- Convolutional Layer 2: 16 filters of size 5x5, followed by ReLU activation.
- Pooling Layer 2: 2x2 average pooling.
- Fully Connected Layer 1: 120 neurons with ReLU activation.
- Fully Connected Layer 2: 84 neurons with ReLU activation.
- Output Layer: 10 neurons (one for each digit class).


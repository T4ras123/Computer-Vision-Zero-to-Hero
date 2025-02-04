# LeNet-5

## Historical Context
LeNet-5, designed by Yann LeCun et al. in 1998, is among the earliest convolutional neural networks. Initially developed for handwritten digit recognition (MNIST), it laid the foundation for modern deep learning-based vision systems.

## Architectural Overview
- **Convolution Layers:** Two convolution layers with $5 \times 5$ filters extract basic visual features.
- **Pooling Layers:** Average pooling reduces spatial dimensions, preserving translation invariance.
- **Fully-Connected Layers:** The network flattens the feature maps and passes them through three linear layers for classification.
- **Activation:** Uses ReLU activations after each convolution and fully connected layer.

## Advantages
- **Simplicity:** Straightforward architecture ideal for beginners.
- **Lightweight:** Contains a low number of parameters, which makes training on modest hardware possible.
- **Historical Significance:** Serves as a starting point for understanding CNN evolution.

## Disadvantages
- **Limited Capacity:** Not suitable for complex image tasks or large datasets.
- **Lack of Modern Techniques:** Does not incorporate techniques like dropout, batch normalization, or deeper layer stacking found in advanced models.

## Other Considerations
LeNet-5 remains an excellent learning tool. Its structure demonstrates key CNN operations such as convolution, pooling, and fully connected layers, ideal for educational purposes.
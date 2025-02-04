# AlexNet

## Historical Context
AlexNet, introduced by Krizhevsky et al. in 2012, sparked the deep learning revolution by winning the ImageNet competition with a wide margin. Its success demonstrated the power of deep convolutional networks on large-scale image recognition tasks.

## Architectural Overview
- **Deep Convolutional Layers:** Consists of five convolutional layers with varying kernel sizes (e.g., $11 \times 11$ and $3 \times 3$) to capture diverse features.
- **ReLU Activation:** Uses ReLU throughout to introduce non-linearity and speed up training.
- **Dropout Layers:** Incorporated before the fully connected layers to reduce overfitting.
- **Softmax Output:** Applies softmax activation to yield class probabilities.
- **Pool Layers:** Max pooling layers reduce spatial dimensions and enhance robustness.

## Advantages
- **Breakthrough Performance:** Pioneered the deep CNN era with significant improvements on benchmark datasets.
- **Effective Regularization:** The use of dropout and data augmentation helps in reducing overfitting.
- **Simplicity yet Depth:** Balances depth with computational efficiency compared to previous architectures.

## Disadvantages
- **High Computational Cost:** Requires considerable computational resources (GPUs) for training.
- **Large Number of Parameters:** Increases risk of overfitting on smaller datasets without proper regularization.
- **Rigid Structure:** Less flexible than modular architectures like GoogLeNet.

## Other Considerations
AlexNet’s architecture has been highly influential in designing subsequent models.
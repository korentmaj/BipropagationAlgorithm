# Bipropagation Algorithm

## Overview
Bipropagation is an innovative neural network training algorithm designed for efficiently training deep neural networks, particularly effective in digit classification tasks. This algorithm adopts a unique layer-by-layer training approach, significantly enhancing the learning process's speed and effectiveness.

## Key Features
- **Layer-by-Layer Training**: Each layer of the neural network is trained individually, allowing for more nuanced adjustments and efficient learning.
- **Initial Weight Optimization**: Starts with weights close to an identity matrix, ensuring a small initial learning error and faster convergence.
- **Versatility**: Suitable for complex classification problems, especially image-based digit recognition.
- **Adaptive Learning**: Adjusts the training process according to the specific characteristics of each input pattern's class.

## Algorithm Workflow
1. **Data Preparation**: Digit images are processed and transformed into a matrix format, suitable for neural network training.
2. **Individual Layer Training**: Begins with perceptron layers, progressively training each one using the bipropagation method.
3. **Corrections and Adjustments**: Applies class-specific corrections to the training data at each layer.
4. **Layer Configuration**: Configures each layer with specific functions (`satlin`, `softmax`) and training parameters (`epochs`, `min_grad`).
5. **Final Network Assembly**: Combines trained layers into a final deep neural network.
6. **Fine-Tuning**: Further trains the assembled network for optimal performance.
7. **Visualization and Evaluation**: Features visualization tools for learned weights and performance evaluation using confusion matrices.

## Technical Overview

The bipropagation algorithm represents a paradigm shift in training deep neural networks, particularly addressing the challenges of vanishing gradients and overfitting in traditional backpropagation. By initializing the weight matrices close to the identity matrix, it ensures minimal initial representation distortion, thereby reducing the initial propagation error and accelerating convergence. The algorithm uniquely applies class-specific perturbations at each layer, enabling refined discriminative feature learning with enhanced generalization capabilities. Each layer's training is decoupled, adopting a greedy layer-wise optimization strategy, reminiscent of deep belief networks, but without the necessity of unsupervised pre-training. The utilization of the `satlin` transfer function, a variant of the sigmoid function with saturation linear characteristics, further stabilizes the learning process, especially in the deeper layers of the network. This approach allows for effective gradient propagation and feature abstraction in deeper architectures, overcoming common pitfalls of gradient-based learning methods. Moreover, the integration of a softmax layer for the final classification stage, trained on the outputs of the preceding perceptron layers, aligns well with the cross-entropy loss minimization, offering a robust framework for tackling complex high-dimensional data such as image pixels in digit classification tasks.


## Contributions
Contributions to the bipropagation algorithm are welcome. Feel free to fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE.md).

## Acknowledgments
- Heavily inspired by the work of Prof. Bojan Ploj Phd on neural network optimization and training techniques.
- Developed by Maj Korent

*This is a brief overview of the bipropagation algorithm. For detailed information and technical specifications, please refer to the accompanying documentation and research papers.*

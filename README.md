# PolyKAN
Testing Different Implementations of Orthogonal KAN
This project explores the integration of various orthogonal polynomial basis functions within a Kolmogorov-Arnold Network (KAN) framework. The goal is to provide a general method that can handle multiple polynomial types and evaluate their effectiveness in terms of accuracy, efficiency, and performance.

## Features

- **Generalized Polynomial Basis Wrapping**: A method to wrap all options and accept different orthogonal polynomial basis functions as arguments within the KAN framework. This allows for flexible integration of multiple orthogonal polynomial functions.
- **Performance Testing**: Evaluation of the chosen polynomial functions based on:
  - Accuracy
  - Number of parameters
  - Time required for completion

### Datasets

The performance of different polynomial bases will be tested on:
- **MNIST Dataset**: A benchmark dataset for digit recognition tasks.
- **Interpolation Tasks**: Testing the interpolation capabilities of each polynomial basis.

## To-Do List

- [x] Implement a general method to wrap different orthogonal polynomial basis functions within KAN.
- [x] Test the implementation based on accuracy, parameter count, and completion time:
    - [x] Using the **MNIST** dataset.
    - [x] On **interpolation** tasks.
- [x] Add Different Layer Combining Functions


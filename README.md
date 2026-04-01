# InductiveVision: CNNs vs. ViTs on ASL MNIST

This repository explores the inductive biases of Convolutional Neural Networks (CNNs) versus Vision Transformers (ViTs) using the American Sign Language (ASL) MNIST dataset. It systematically benchmarks how both architectures perform under varying data availability and visualizes their learned representations (local edge detectors vs. global self-attention).

## 🚀 Project Overview

While CNNs rely on strong local inductive biases (translations, local receptive fields), Vision Transformers rely on global self-attention mechanisms that generally require more data to learn visual structure from scratch. This project investigates these differences through:
- **Benchmarking**: Evaluating both models on 1%, 5%, 10%, 50%, and 100% of the training data limits.
- **Data Augmentation**: Assessing the impact of random rotations and crops to simulate real-world variance.
- **Visualization**: Extracting CNN early-layer feature maps and ViT attention maps to compare feature learning.
- **Error Analysis**: Generating detailed confusion matrices to track common misclassifications among ASL characters.

## 🗂️ Repository Structure

- `benchmark_models.py`: Main script to train and evaluate CNN and ViT models across different dataset percentiles and augmentation settings. Saves metrics to `benchmark_results.json`.
- `plot_benchmarks.py`: Parses the benchmark results and plots accuracy vs. data size, loss vs. data size, and theoretical receptive fields.
- `error_analysis.py`: Evaluates the fully trained CNN and ViT on the test set, generating comparative confusion matrices.
- `eda_and_augmentation.py`: Automates Exploratory Data Analysis (EDA) and saves plots for class distributions, pixel variances, and data augmentation comparisons.
- `visualize_models.py`: Trains lightweight versions of both models to extract and visualize CNN convolutional filters (edges) and ViT self-attention grids.
- `model.ipynb` / `Transformer_attempt.ipynb`: Jupyter notebooks preserving the standalone pipeline implementations of `ASLResNet` and `SignLanguageViT`.

## ⚙️ Installation & Requirements

Ensure you have Python 3.8+ installed. The primary dependencies are:
- `torch` & `torchvision` (PyTorch)
- `pandas` & `numpy`
- `matplotlib` & `seaborn`
- `scikit-learn`

Install the required packages using pip:
```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn
```

## 🛠️ Usage

1. **Exploratory Data Analysis**
   Run the EDA script to visualize class distributions and pixel variance:
   ```bash
   python eda_and_augmentation.py
   ```
2. **Run the Full Benchmarks**
   Compare the CNN and ViT models across sub-sampled datasets:
   ```bash
   python benchmark_models.py
   ```
3. **Generate Benchmark Plots & Tables**
   ```bash
   python plot_benchmarks.py
   ```
4. **Visualize Attention & Convolutional Edges**
   ```bash
   python visualize_models.py
   ```
5. **Evaluate Final Errors**
   ```bash
   python error_analysis.py
   ```

*Outputs, including plots, confusion matrices, and performance markup tables, are automatically saved to the `benchmark_plots/` and `output_plots/` directories.*

## 📊 Dataset

The models are trained using the **Sign Language MNIST** dataset, which consists of 28x28 grayscale images of hands forming the letters of the alphabet (A-Z, excluding J and Z due to motion). 

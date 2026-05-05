# Federated Learning Client Drift Analysis

This project implements a controlled Federated Learning experiment to study how increasing label skew affects client drift, model stability, and global model accuracy.

The experiment uses the MNIST dataset, a CNN model, and the Federated Averaging (FedAvg) algorithm across 10 simulated clients. Three levels of non-IID label distribution are tested: low skew, medium skew, and high skew.

## Objective

Most federated learning experiments measure only final accuracy. This project goes deeper by tracking:

- Global model accuracy
- Client drift
- Model stability

The goal is to show that accuracy alone may hide internal instability in federated systems.

## Key Findings

- High label skew reduces final model accuracy.
- Client drift increases significantly as label skew increases.
- Model stability worsens under high skew.
- Drift and stability act as early warning signals before accuracy degradation becomes obvious.

In the experiment, high label skew increased client drift by approximately 4.1× and model instability by approximately 6.0× compared to low skew.

## Tech Stack

- Python
- TensorFlow
- NumPy
- Matplotlib
- MNIST Dataset
- Jupyter Notebook

## Methodology

The experiment follows a standard federated learning setup:

1. Load MNIST dataset.
2. Split data across 10 simulated clients.
3. Create three skew settings:
   - Low skew
   - Medium skew
   - High skew
4. Train local CNN models on each client.
5. Aggregate model weights using FedAvg.
6. Measure:
   - Accuracy after aggregation
   - Client drift before aggregation
   - Rolling model stability across rounds

## Federated Learning Setup

- Dataset: MNIST
- Number of clients: 10
- Algorithm: Federated Averaging (FedAvg)
- Model: Convolutional Neural Network
- Communication rounds: 50
- Local epochs per round: 5
- Optimizer: Adam
- Batch size: 32

## Metrics

### Accuracy

Measures the global model’s classification performance on the MNIST test set.

### Client Drift

Measures the average pairwise distance between client model weights after local training and before aggregation.

### Model Stability

Measures round-to-round fluctuation in global accuracy using rolling standard deviation.

## Project Structure

```text
federated-learning-client-drift/
│
├── notebooks/
│   └── fedavg_label_skew_experiment.ipynb
│
├── reports/
│   └── seminar_report.pdf
│
├── results/
│   ├── accuracy_drift_plot.png
│   ├── mean_drift_accuracy_plot.png
│   ├── stability_plot.png
│   └── skew_heatmaps.png
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE

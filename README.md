# CIFAR10_Resnet18

Train a PyTorch model on CIFAR10 dataset. Apply Post Training Quantization on the model using Dynamic Quantization.

Evaluating and comparing metrics for the trained model and quantized model.

### Setting Up the environment

Creating a virtual environment using Conda

```bash
conda create -n torch
conda activate torch
```

Installing dependencies
```bash
pip install -r requirements.txt
```

### Run the quantization code

```bash
python3 dyn_quant_cifar.py
```

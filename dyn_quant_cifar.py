import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import time
import copy
import numpy as np

from resnet import resnet18

import torch.nn.utils.prune as prune


def set_random_seeds(random_seed=0):
    '''
    Setting default random seeds

    '''

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def prepare_dataloader(num_workers=8,
                       train_batch_size=128,
                       eval_batch_size=256):

    '''
    Data Transformations on CIFAR10 datasets from torchvision.datasets library
    Returns Train and Test DataLoaders

    '''

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    train_set = torchvision.datasets.CIFAR10(root="data",
                                             train=True,
                                             download=True,
                                             transform=train_transform)
   
    test_set = torchvision.datasets.CIFAR10(root="data",
                                            train=False,
                                            download=True,
                                            transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=train_batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=eval_batch_size,
                                              sampler=test_sampler,
                                              num_workers=num_workers)

    return train_loader, test_loader


def evaluate_model(model, test_loader, criterion=None):
    '''
    Predicting outputs and evaluating the model
    Returns evaluation loss and evaluation accuracy

    '''

    model.eval()

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:


        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy


def train_model(model,
                train_loader,
                test_loader,
                learning_rate=1e-1,
                num_epochs=10):
    '''
    Train the Model and calculate training loss and accuracy
    Returns the trained model

    '''

    criterion = nn.CrossEntropyLoss()


    # SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=0.9,
                          weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[100, 150],
                                                     gamma=0.1,
                                                     last_epoch=-1)

    model.eval()
    eval_loss, eval_accuracy = evaluate_model(model=model,
                                              test_loader=test_loader,
                                              criterion=criterion)
    print("Epoch: {:02d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
        -1, eval_loss, eval_accuracy))

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        model.eval()
        eval_loss, eval_accuracy = evaluate_model(model=model,
                                                  test_loader=test_loader,
                                                  criterion=criterion)

        scheduler.step()

        print(
            "Epoch: {:03d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}"
            .format(epoch, train_loss, train_accuracy, eval_loss,
                    eval_accuracy))

    return model

def measure_inference_latency(model,
                              input_size=(1, 3, 32, 32),
                              num_samples=100,
                              num_warmups=10):
    '''
    Evaluates the model and calculates inference time
    Returns time latency for inference of the model

    '''

 
    model.eval()

    x = torch.rand(size=input_size)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave


def save_model(model, model_dir, model_filename):
    '''
    Save the trained model

    '''

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)

def load_model(model, model_filepath):
    '''
    Load the model from the saved model path

    '''

    model.load_state_dict(torch.load(model_filepath))

    return model

def save_torchscript_model(model, model_dir, model_filename):
    '''
    Saving torchscript version of trained model
    
    '''

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)

def load_torchscript_model(model_filepath):
    '''
    Load the saved torchscript model from saved model path

    '''

    model = torch.jit.load(model_filepath)

    return model


def create_model(num_classes=10):
    '''
    Create the initial model using custom resnet18 configuration from resnet.py file

    '''
    
    model = resnet18(num_classes=num_classes, pretrained=False)
    return model

def print_size_of_model(model, label=""):
    '''
    Print the size of the model

    '''
    
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

def main():

    random_seed = 0
    num_classes = 10

    model_dir = "saved_models"
    model_filename = "resnet18_cifar10.pt"
    quantized_model_filename = "resnet18_quantized_cifar10.pt"
    model_filepath = os.path.join(model_dir, model_filename)
    quantized_model_filepath = os.path.join(model_dir,
                                            quantized_model_filename)

    set_random_seeds(random_seed=random_seed)

    # Create an initial untrained model using custom resnet18 configuration
    model = create_model(num_classes=num_classes)

    train_loader, test_loader = prepare_dataloader(num_workers=8,
                                                   train_batch_size=128,
                                                   eval_batch_size=256)

    # Train the model
    model = train_model(model=model,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        learning_rate=1e-1,
                        num_epochs=10)
    # Save the trained model.
    save_model(model=model, model_dir=model_dir, model_filename=model_filename)
    
    # Load a trained model.
    model = load_model(model=model,
                       model_filepath=model_filepath)
    
    # Create a model copy for layer fusion and quanitization will be performed on the fused model
    fused_model = copy.deepcopy(model)

    # Set both models to evaluation mode for quantization to work
    model.eval()
    fused_model.eval()

    # Fuse the model in place rather manually.
    fused_model = torch.quantization.fuse_modules(fused_model,
                                                  [["conv1", "bn1", "relu"]],
                                                  inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(
                    basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]],
                    inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block,
                                                        [["0", "1"]],
                                                        inplace=True)

    # Print FP32 model.
    print('FP32 Model: \n',model)

    # Print fused model.
    print('Fused Model: \n',fused_model)

    # Quantize the model using Dynamic Quantization
    torch.backends.quantized.engine = 'qnnpack'
    quantized_model = torch.quantization.quantize_dynamic(
        fused_model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8)
    
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    quantized_model.qconfig = quantization_config

    # Print quantization configurations
    print(quantized_model.qconfig)

    quantized_model = torch.quantization.convert(quantized_model, inplace=True)

    quantized_model.eval()

    # Print quantized model.
    print('Quantized int8 Model: \n',quantized_model)

    # Save quantized model.
    save_torchscript_model(model=quantized_model,
                           model_dir=model_dir,
                           model_filename=quantized_model_filename)

    # Load quantized model.
    quantized_jit_model = load_torchscript_model(
        model_filepath=quantized_model_filepath)

    # Evaluate and Gain Inference on both fp32 and quantized models
    _, fp32_eval_accuracy = evaluate_model(model=model,
                                           test_loader=test_loader,
                                           criterion=None)
    _, int8_eval_accuracy = evaluate_model(model=quantized_jit_model,
                                           test_loader=test_loader,
                                           criterion=None)

    print("FP32 evaluation accuracy: {:.3f}".format(fp32_eval_accuracy))
    print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))

    # Calculate inference time latency for the models 
    fp32_cpu_inference_latency = measure_inference_latency(model=model,
                                                           input_size=(1, 3,
                                                                       32, 32),
                                                           num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(
        model=quantized_model,
        input_size=(1, 3, 32, 32),
        num_samples=100)
    
    int8_jit_cpu_inference_latency = measure_inference_latency(
        model=quantized_jit_model,
        input_size=(1, 3, 32, 32),
        num_samples=100)
    
    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(
        fp32_cpu_inference_latency * 1000))
    print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(
        int8_cpu_inference_latency * 1000))
    print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(
        int8_jit_cpu_inference_latency * 1000))
    
    # Calculate and compare size of fp32 and quantized models
    f=print_size_of_model(model,"fp32")
    q=print_size_of_model(quantized_model,"int8")
    print("{0:.6f} times smaller".format(f/q))

if __name__ == "__main__":

    main()
import os
import tarfile
import time
from pathlib import Path
from typing import Literal
import pandas as pd
import numpy as np
import pytorch_quantization
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torchdata
import torch_tensorrt
import torch_tensorrt as torchtrt
import torchvision.transforms as transforms
import wget
from modelopt.torch.quantization.utils import export_torch_mode
from torchvision import datasets, models

cudnn.benchmark = True

print(pytorch_quantization.__version__)


# Define main data directory
DATA_DIR = Path.home() / "./data/imagenette2-320"
# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")


def download_data(DATA_DIR):
    """
    Downloads a subset of imagenet with 10 classes
    'tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball' and 'parachute'.
    """
    if os.path.exists(DATA_DIR):
        if not os.path.exists(os.path.join(DATA_DIR, "imagenette2-320")):
            url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
            wget.download(url)
            # open file
            file = tarfile.open("imagenette2-320.tgz")
            # extracting file
            file.extractall(DATA_DIR)
            file.close()
            os.remove("imagenette2-320.tgz")
    else:
        print("This directory doesn't exist. Create the directory and run again")


if not os.path.exists(Path.home() / "data"):
    os.mkdir(Path.home() / "data")
download_data(Path.home() / "data")


# This function allows you to set the all the parameters to not have gradients,
# allowing you to freeze the model and not undergo training during the train step.
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# Define functions for training, evalution, saving checkpoint and train parameter setting function
def train(model: nn.Module, dataloader: torchdata.DataLoader, crit, opt, epoch: int):
    model.train()
    running_loss = 0.0
    for idx, (batch, labels) in enumerate(dataloader):
        batch, labels = batch.cuda(), labels.cuda(non_blocking=True)
        opt.zero_grad()
        out = model(batch)
        loss = crit(out, labels)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        if idx % 100 == 99:
            print(
                "Batch: [%5d | %5d] loss: %.3f"
                % (idx + 1, len(dataloader), running_loss / 100)
            )
            running_loss = 0.0


def dtype_to_torch_type(dtype: Literal["fp32", "fp16", "int8"]):
    match dtype:
        case "fp32":
            return torch.float32
        case "fp16":
            return torch.float16
        case "int8":
            return torch.int8
        case _:
            assert False


def evaluate(model, dataloader, crit, epoch: int, dtype: Literal["fp32", "fp16"]):
    total = 0
    correct = 0
    loss = 0.0
    class_probs = []
    class_preds = []
    model.eval()

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.cuda(), labels.cuda(non_blocking=True)
            out = model(data.to(dtype_to_torch_type(dtype)))
            loss += crit(out, labels)
            preds = torch.max(out, 1)[1]
            class_probs.append([F.softmax(i, dim=0) for i in out])
            class_preds.append(preds)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    evaluate_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    evaluate_preds = torch.cat(class_preds)

    return loss / total, correct / total


def save_checkpoint(state, ckpt_path="checkpoint.pth"):
    torch.save(state, ckpt_path)
    print("Checkpoint saved")


# Helper function to benchmark the model
def timeit(model,dtype: Literal["fp32", "fp16"], input_shape=(1024, 1, 32, 32), nwarmup=10, nruns=10):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda").to(dtype_to_torch_type(dtype))

    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            _ = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
    print("Average batch time: %.2f ms" % (np.mean(timings) * 1000))
    return np.mean(timings) * 1000


def dataset_splits():
    # Performing Transformations on the dataset and defining training and validation dataloaders
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    calib_dataset = torch.utils.data.random_split(val_dataset, [2901, 1024])[1]
    train_dataloader = torchdata.DataLoader(
        train_dataset, batch_size=32, shuffle=True, drop_last=True
    )
    val_dataloader = torchdata.DataLoader(
        val_dataset, batch_size=64, shuffle=False, drop_last=True
    )
    calib_dataloader = torchdata.DataLoader(
        calib_dataset, batch_size=64, shuffle=False, drop_last=True
    )
    return train_dataloader, val_dataloader, calib_dataloader


def load_model(feature_extract: bool):
    model = models.mobilenet_v2(pretrained=True)
    set_parameter_requires_grad(model, feature_extracting=feature_extract)
    model.classifier[1] = nn.Linear(1280, 10)
    model = model.cuda()
    return model


def main():
    train_dataloader, val_dataloader, calib_dataloader = dataset_splits()
    # we set all requires grad to false on the backbone, then train a new classifier
    model = load_model(feature_extract=True)
    # Define a classification head for 10 classes.
    # Declare Learning rate
    lr = 0.0001

    # Use cross entropy loss for classification and SGD optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Train the model for 3 epochs to attain an acceptable accuracy.
    num_epochs = 3
    for epoch in range(num_epochs):
        print("Epoch: [%5d / %5d] LR: %f" % (epoch + 1, num_epochs, lr))

        train(model, train_dataloader, criterion, optimizer, epoch)
        test_loss, test_acc = evaluate(model, val_dataloader, criterion, epoch)

        print("Test Loss: {:.5f} Test Acc: {:.2f}%".format(test_loss, 100 * test_acc))

    save_checkpoint(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "acc": test_acc,
            "opt_state_dict": optimizer.state_dict(),
        },
        ckpt_path="mobilenetv2_base_ckpt",
    )


def benchmark(
    method_name: str, model, val_dataloader, criterion, dtype: Literal["fp32", "fp16"]
) -> dict:
    test_loss, test_acc = evaluate(model, val_dataloader, criterion, 0, dtype=dtype)
    timings = timeit(model, input_shape=(64, 3, 224, 224), dtype=dtype)
    print(f"{method_name}: {(test_acc * 100):.2f}%")
    return {"method": method_name, "time": timings, "acc": float(test_acc), "loss": float(test_loss)}


def compile():
    train_dataloader, val_dataloader, _ = dataset_splits()
    res = []
    
    model = load_model(False)
    model.load_state_dict(torch.load("mobilenetv2_base_ckpt")["model_state_dict"])
    
    logs = benchmark("loaded_baseline_model-fp32", model, val_dataloader, nn.CrossEntropyLoss(), "fp32")
    res.append(logs)

    with torch.no_grad():
        images, _ = next(iter(val_dataloader))
        jit_model = torch.jit.trace(model, images.to("cuda"))
        torch.jit.save(jit_model, "mobilenetv2_base.jit.pt")
   
    logs = benchmark("jit_traced-fp32", jit_model, val_dataloader, nn.CrossEntropyLoss(), "fp32")
    res.append(logs)
    
    # Loading the Torchscript model and compiling it into a TensorRT model
    baseline_model = torch.jit.load("mobilenetv2_base.jit.pt").eval()

    logs = benchmark("torch.jit.loaded", baseline_model, val_dataloader, nn.CrossEntropyLoss(), "fp32")

    compile_spec = {
        "inputs": [torch_tensorrt.Input([64, 3, 224, 224])],
        "enabled_precisions": torch.float,
    }
    trt_base = torch_tensorrt.compile(baseline_model, **compile_spec)
    logs = benchmark("tensorrt.compiled-fp32", trt_base, val_dataloader, nn.CrossEntropyLoss(), "fp32")
    res.append(logs)

    # compile with torch
    backend_kwargs = {
        "enabled_precisions": {torch.half},  # quantization
        "min_block_size": 2,  # for larger blocks more ops fall unto python
        "torch_executed_ops": {
            "torch.ops.aten.sub.Tensor"
        },  # some operations cannot by compiled, they stay in pytorch
        "optimization_level": 4,  # [0-4] higher means for aggressive fusion
        "use_python_runtime": False,  # fully executed in tensor rt
    }
    # here I don't have to run torch.jit.trace?
    # Run the model on an input to cause compilation, as so:
    model = torch.compile(
        model.half().cuda(),
        backend="torch_tensorrt",  # uses the dynamo frontend to trace the model, here you don't need torch.jit.compile
        options=backend_kwargs,
        dynamic=False,  # input shape is static
    )
    with torch.no_grad():
        model(torch.randn((64, 3, 224, 224), device="cuda", dtype=torch.half))

    save_checkpoint({"model_state_dict": model.state_dict()}, "compiled_mobilenet_ckpt")
    
    logs = benchmark("torch.compile-fp16", model, val_dataloader, nn.CrossEntropyLoss(), "fp16")
    res.append(logs)

    """
    I tested this with fp8 and int8;
    for fp8 one operation was not implemented in cuda
    int8 it complains that the nn.Module only accepts floats
    """
    model = load_model(False)
    _, val_dataloader, _ = dataset_splits()
    images, _ = next(iter(val_dataloader))

    dtype= "fp16"
    torch_type = torch.half
    enabled_precisions = {torch.half}
    logs = benchmark("baseline_model-fp16", model.to(torch_type), val_dataloader, nn.CrossEntropyLoss(), "fp16")
    res.append(logs)

    with torch.no_grad():
        with export_torch_mode():
            # Compile the model with Torch-TensorRT Dynamo backend

            # does not seem to work without explicitly casting the model to the right type
            input_tensor = images.cuda()
            exp_program = torch.export.export(
                model.to(torch_type), (input_tensor.to(torch_type),), strict=False
            )
            trt_model = torchtrt.dynamo.compile(
                exp_program,
                inputs=[input_tensor],
                enabled_precisions=enabled_precisions,
                min_block_size=1,
            )
            logs = benchmark(
                f"torch_export+dynamo-{torch_type}",
                trt_model,
                val_dataloader,
                nn.CrossEntropyLoss(),
                dtype,
            )
            res.append(logs)
            logs = benchmark(
                f"torch-modelin{dtype}",
                model.to(torch_type),
                val_dataloader,
                nn.CrossEntropyLoss(),
                dtype,
            )
            res.append(logs)

        # You can also use torch compile path to compile the model with Torch-TensorRT:
        # trt_model = torch.compile(model, backend="tensorrt")
    df = pd.DataFrame(res)
    df.to_csv("benchmarking.csv")
    print(df.to_markdown())


def quant_aware_training():
    # We define Mobilenetv2 again just like we did above
    # All the regular conv, FC layers will be converted to their quantized counterparts due to quant_modules.initialize()
    feature_extract = False
    q_model = models.mobilenet_v2(pretrained=True)
    set_parameter_requires_grad(q_model, feature_extract)
    q_model.classifier[1] = nn.Linear(1280, 10)
    q_model = q_model.cuda()

    # mobilenetv2_base_ckpt is the checkpoint generated from Step 2 : Training a baseline Mobilenetv2 model.
    ckpt = torch.load("./mobilenetv2_base_ckpt")
    modified_state_dict = {}
    for key, val in ckpt["model_state_dict"].items():
        # Remove 'module.' from the key names
        if key.startswith("module"):
            modified_state_dict[key[7:]] = val
        else:
            modified_state_dict[key] = val

    # Load the pre-trained checkpoint
    q_model.load_state_dict(modified_state_dict)
    optimizer = optim.SGD(
        q_model.parameters(), lr=0.1
    )  # parameters should not matter much since they are loaded

    optimizer.load_state_dict(ckpt["opt_state_dict"])

def read():
    print(pd.read_csv("benchmarking.csv").to_markdown())
# main()
# compile()
read()
compile()

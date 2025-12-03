import pandas as pd
import pytorch_quantization
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch_tensorrt
import torch_tensorrt as torchtrt
from modelopt.torch.quantization.utils import export_torch_mode

cudnn.benchmark = True

print(pytorch_quantization.__version__)


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


def compile():
    train_dataloader, val_dataloader, _ = dataset_splits()
    res = []

    model = load_model(False)
    model.load_state_dict(torch.load("mobilenetv2_base_ckpt")["model_state_dict"])

    logs = benchmark(
        "loaded_baseline_model-fp32",
        model,
        val_dataloader,
        nn.CrossEntropyLoss(),
        "fp32",
    )
    res.append(logs)

    with torch.no_grad():
        images, _ = next(iter(val_dataloader))
        jit_model = torch.jit.trace(model, images.to("cuda"))
        torch.jit.save(jit_model, "mobilenetv2_base.jit.pt")

    logs = benchmark(
        "jit_traced-fp32", jit_model, val_dataloader, nn.CrossEntropyLoss(), "fp32"
    )
    res.append(logs)

    # Loading the Torchscript model and compiling it into a TensorRT model
    baseline_model = torch.jit.load("mobilenetv2_base.jit.pt").eval()

    logs = benchmark(
        "torch.jit.loaded",
        baseline_model,
        val_dataloader,
        nn.CrossEntropyLoss(),
        "fp32",
    )

    compile_spec = {
        "inputs": [torch_tensorrt.Input([64, 3, 224, 224])],
        "enabled_precisions": torch.float,
    }
    trt_base = torch_tensorrt.compile(baseline_model, **compile_spec)
    logs = benchmark(
        "tensorrt.compiled-fp32",
        trt_base,
        val_dataloader,
        nn.CrossEntropyLoss(),
        "fp32",
    )
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

    logs = benchmark(
        "torch.compile-fp16", model, val_dataloader, nn.CrossEntropyLoss(), "fp16"
    )
    res.append(logs)

    """
    I tested this with fp8 and int8;
    for fp8 one operation was not implemented in cuda
    int8 it complains that the nn.Module only accepts floats
    """
    model = load_model(False)
    _, val_dataloader, _ = dataset_splits()
    images, _ = next(iter(val_dataloader))

    dtype = "fp16"
    torch_type = torch.half
    enabled_precisions = {torch.half}
    logs = benchmark(
        "baseline_model-fp16",
        model.to(torch_type),
        val_dataloader,
        nn.CrossEntropyLoss(),
        "fp16",
    )
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


def read():
    print(pd.read_csv("benchmarking.csv").to_markdown())


# main()
# compile()
read()
compile()

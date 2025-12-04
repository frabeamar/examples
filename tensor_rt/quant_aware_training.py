import pandas as pd
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, assert_never

import torch
import torch.nn as nn
import torch_tensorrt
from pytorch_quantization import calib, quant_modules
from pytorch_quantization import nn as quant_nn
from tqdm import tqdm

from main import benchmark
from train import dataset_splits, evaluate, load_model, save_checkpoint, train_epoch


def compute_amax(model: nn.Module, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(f"{name:40}: {module}")
    model.cuda()


def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


@dataclass
class MaxCalib:
    pass


@dataclass(frozen=True)
class HistogramCalib:
    method: Literal["mse", "entropy", "percentile"]
    percentile: float


def calibrate_model(
    model,
    model_name,
    data_loader,
    num_calib_batch: int,
    calibrator: MaxCalib | HistogramCalib,
    out_dir,
):
    """
    Feed data to the network and calibrate.
    Arguments:
        model: classification model
        model_name: name to use when creating state files
        data_loader: calibration data set
        num_calib_batch: amount of calibration passes to perform
        calibrator: type of calibration to use (max/histogram)
        hist_percentile: percentiles to be used for historgram calibration
        out_dir: dir to save state files in
    """

    print("Calibrating model")
    with torch.no_grad():
        collect_stats(model, data_loader, num_calib_batch)
    match calibrator:
        case HistogramCalib(method="percentile", percentile=prc):
            compute_amax(model, method="percentile")
            calib_output = os.path.join(
                out_dir,
                f"{model_name}-percentile-{prc}-{num_calib_batch * data_loader.batch_size}.pth",
            )
            torch.save(model.state_dict(), calib_output)
        case HistogramCalib(method=method, percentile=prc):
            compute_amax(model, method=method)
            calib_output = os.path.join(
                out_dir,
                f"{model_name}-{method}-{num_calib_batch * data_loader.batch_size}.pth",
            )
            torch.save(model.state_dict(), calib_output)

        case MaxCalib():
            compute_amax(model, method="max")
            calib_output = os.path.join(
                out_dir,
                f"{model_name}-max-{num_calib_batch * data_loader.batch_size}.pth",
            )
            torch.save(model.state_dict(), calib_output)
        case _:
            assert_never(calibrator)

    return calib_output


def quant_aware_calibration():
    quant_modules.initialize()
    _, _, calib_dataloader = dataset_splits()
    qat_model = load_model(False)
    res = []
    # Calibrate the model using max calibration technique.
    with torch.no_grad():
        for calibrator in [MaxCalib()] + [
            HistogramCalib("mse", 0.99),
            HistogramCalib("entropy", 0.99),
            HistogramCalib("percentile", 0.99),
        ]:
            print(calibrator)
            calib_model_path = calibrate_model(
                model=qat_model,
                model_name="mobilenet",
                data_loader=calib_dataloader,
                num_calib_batch=32,
                calibrator=calibrator,
                out_dir="./",
            )


state = {}


# Adjust learning rate based on epoch number
def adjust_lr(lr: float, optimizer, epoch):
    global state
    new_lr = lr * (0.5 ** (epoch // 12)) if state["lr"] > 1e-7 else state["lr"]
    if new_lr != state["lr"]:
        state["lr"] = new_lr
        print("Updating learning rate: {}".format(state["lr"]))
        for param_group in optimizer.param_groups:
            param_group["lr"] = state["lr"]


def finetune_with_quantized(
    model, train_dataloader, eval_dataloader, optimizer, num_epochs: int
):
    """
    After getting the bounds for quantization you still have to finetune the weights
    This should be with an anneelead learning rate and 10% of the training time.

    """
    for epoch in range(num_epochs):
        train_epoch(model, train_dataloader, optimizer)
        test_loss, test_acc = evaluate(model, eval_dataloader)
        print("Test Loss: {:.5f} Test Acc: {:.2f}%".format(test_loss, 100 * test_acc))
    return model

    # save_checkpoint({'epoch': epoch + 1,
    #                 'model_state_dict': model.state_dict(),
    #                 'acc': test_acc,
    #                 'opt_state_dict': optimizer.state_dict()},
    #                 ckpt_path="mobilenet-finetuned")


def quant_aware_train():
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    train_dataloader, eval_dataloader, calib_dataloader = dataset_splits()
    model = load_model(False)
    for m in glob.glob("*.pth"):
        # here strict is False as we want to ingore the quantizer max elements
        model.load_state_dict(torch.load(m), strict=False)
        model = finetune_with_quantized(
            model,
            train_dataloader,
            eval_dataloader,
            torch.optim.SGD(model.parameters(), lr=0.001),
            1,
        )
        save_checkpoint({"model_state_dict": model.state_dict()}, Path(m).stem + "_finetuned.pth")


def compile():
    res = []
    train_dataloader, eval_dataloader, calib_dataloader = dataset_splits()
    for m in glob.glob("*finetuned.pth"):
        model = load_model(False).eval()
        model.load_state_dict(torch.load(m)["model_state_dict"], strict=True)
        with torch.no_grad():
            data = iter(calib_dataloader)
            images, _ = next(data)
            # error with gratest relative difference:
            # if the model is fake quantized jit cannot reproduce them
            # you should finetune first
            jit_model = torch.jit.trace(model, images.to("cuda"))
            torch.jit.save(jit_model, Path(m).stem + "traced.pt")

        compile_spec = {
            "inputs": [torch_tensorrt.Input([64, 3, 224, 224])],
            "enabled_precisions": torch.float,
            # "truncate_long_and_double_enabled"
        }
        # bad_nodes = []
        # for node in jit_model.inlined_graph.nodes():
        #     for out in node.outputs():
        #         t = out.type()
        #         if hasattr(t, "scalarType") and t.scalarType() in ["Long", "Double"]:
        #             bad_nodes.append((node, t))

        trt_mod = torch_tensorrt.compile(jit_model, **compile_spec)
        logs = benchmark(Path(m).stem + "compiled", trt_mod, eval_dataloader)
        res.append(logs)
    df = pd.DataFrame(res)
    df.to_csv("quant_aware_benchmarking.csv")
    


# quant_aware_train()
compile()

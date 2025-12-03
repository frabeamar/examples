import os
from dataclasses import dataclass
from typing import Literal, assert_never

import pandas as pd
import torch
import torch.nn as nn
from pytorch_quantization import calib, quant_modules
from pytorch_quantization import nn as quant_nn
from tqdm import tqdm

from main import benchmark
from train import dataset_splits, load_model


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
            calib_model_path = calibrate_model(
                model=qat_model,
                model_name="mobilenet",
                data_loader=calib_dataloader,
                num_calib_batch=32,
                calibrator=calibrator,
                out_dir="./",
            )
            model = qat_model.load_state_dict(torch.load(calib_model_path))
            logs = benchmark(model, input_shape=(64, 3, 224, 224))
            res.append(logs)

    df = pd.DataFrame(res)
    print(df.to_markdown())
    df.to_csv("quant_aware_benchmarking.csv")


quant_aware_calibration()

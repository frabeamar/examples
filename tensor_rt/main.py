import torch
import torch.backends.cudnn as cudnn
import torch_tensorrt

from train import benchmark

cudnn.benchmark = True


def main():
    """
    Benchmark the two models,
    the compiled version should be
    """
    precision = "fp32"
    ssd300 = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd", model_math=precision
    )

    model = torch.jit.load("trt_model.ts").eval()
    
    benchmark(model,  nwarmup=10, nruns=10)

    model = ssd300.eval().to("cuda")
    benchmark(model, input_shape=(128, 3, 300, 300), nwarmup=10, nruns=10)


def trace_model():
    precision = "fp32"

    ssd300 = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd", model_math=precision
    )
    model = ssd300.eval().to("cuda")
    half_tensor = torch.randn((1, 3, 300, 300), dtype=torch.float32)
    traced_model = torch.jit.trace(model, [half_tensor.to("cuda")])

    # The compiled module will have precision as specified by "op_precision".
    # Here, it will have FP16 precision.
    # needs to be compiled in the same shape as the one that you are going to run it with
    trt_model = torch_tensorrt.compile(
        traced_model,
        inputs=[torch_tensorrt.Input((128, 3, 300, 300), dtype=torch.float32)],
        enabled_precisions={torch.half},  # Run with FP16
        workspace_size=1
        << 20,  # maximum temporary gpu that can be used during optimization.
    )
    torch.jit.save(trt_model, "trt_model.ts")


if __name__ == "__main__":
    trace_model()
    main()

import torch
import time
import numpy as np
import torch_tensorrt

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


# Helper function to benchmark the model
def benchmark(model, input_shape=(1024, 1, 32, 32), dtype='fp32', nwarmup=50, nruns=1000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()
    
    # you need to warm up the GPU:
    # it loads the kernel, compiles them for the right shape and type, allocates memory, fills the cache
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            pred_loc, pred_label  = model(input_data)
            # cpu has to wait that gpu work is done before timing the model 
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print("Output location prediction size:", pred_loc.size())
    print("Output label prediction size:", pred_label.size())
    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))



def main():
    precision = "fp32"
    ssd300 = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd", model_math=precision
    )

    model = torch.jit.load("trt_model.ts")
    benchmark(model, input_shape=(128, 3, 300, 300), nwarmup=10, nruns=10)


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
    trt_model = torch_tensorrt.compile(traced_model,
        inputs= [torch_tensorrt.Input((128, 3, 300, 300), dtype=torch.float32)],
        enabled_precisions= {torch.half}, # Run with FP16
        workspace_size= 1 << 20 # maximum temporary gpu that can be used during optimization.

    )
    torch.jit.save(trt_model, "trt_model.ts")

    loaded_model = torch.jit.load("trt_model.ts")
    loaded_model.eval()  # switch to evaluation mode

trace_model()
main()

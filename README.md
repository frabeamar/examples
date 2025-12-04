# examples
A collection of notes / tutorial and examples. Mostly deeplearning. 
Might not be pretty
## diffusers
trained ddpm via huggingface on cifar. 
## tensor_rt

JIT compiler -> just in time; only the parts that run often get optimize
In some update for pytorch they introduced torch script to make the models compilable, it uses jit (just in time compiler), it is similar to tensorflow graph mode.
This removes the python dependencies that the graph may have
there are two ways to generate a model with tensorRT 
 - tracing  : follow execution of the model, and generate tensor rt ops which replicate the behaviour
 - scripting : analyze the code; this can add control flow, which scripting cannot do.
 Tracing is easier

??? maybe true => ai
Torchscript is being phased out as it has several problems:
 - limited python support
 - criptic messages for debugging
 now you use torch.export to create an itermediate representation, a flattened computational graph with low level op. This can be then compiled to get a platform specific  binary

An alternative to TorchScript is Tensor RT. Instead of being a jit compiler it is an aot (ahead of time) compiler -> you explicitly go through a compile step

torch.compile -> jit compilerl it will fall back to eager execution if somehting untraceble is found
torch.export.export -> aot compiler; it will give an error 
torch.export is expected to work on more user programs, and produce lower-level graphs (at the torch.ops.aten operator level)
if cannot capture control flow like torch.jit.script 

### PTQ: post training quantization
you need calibration data, a subset of train and validation to understand in what range are the values to be quantized. there are three strategies 
 - min-max, can be affected by outliers
 - entropy, minimize the information loss with the fp32 representation (still a mapping to a centroid, but the distance is not squared but controlled on the distribution of values)
 - percentile
 Quantization could cause errors, especially with the depthwise convolution ->  the error message is also slightly different
For mobilenet v2:

|    | method                            |    time |       acc |      loss |
|---:|:----------------------------------|--------:|----------:|----------:|
|  0 | loaded_baseline_model-fp32        | 43.144  | 0.775102  | 0.0254265 |
|  1 | jit_traced-fp32                   | 42.8415 | 0.775102  | 0.0254265 |
|  2 | tensorrt.compiled-fp32            | 18.2159 | 0.775102  | 0.0254265 |
|  3 | torch.compile-fp16                | 16.3828 | 0.775102  | 0.0254517 |
|  4 | baseline_model-fp16               | 24.739  | 0.0404713 | 0.0375671 |
|  5 | torch_export+dynamo-torch.float16 | 12.4445 | 0.0412398 | 0.0375671 |
|  6 | torch-torch.float16               | 24.7175 | 0.0404713 | 0.0375671 |

The base model starts at 77 % accuracy and 43 ms
Tracing the model with jit does not cahnge the time. 
Compiling it with torch.compile or tensorrt compile makes it almost twice as quick (makes sense it is the same background)
Running the model in fp16 takes half the time the accuracy drops to 4%
This drop makes sense for the baseline_model-fp16 but it should not have happend to the torch_export with the dynamo frontend. 
Maybe I set up the tracing wrong 

 ### Quantization aware training
 you model the quantization during training -> after the activation quantize the values and dequantize, so the next block receive fp32 again. During backprop the gradient is undefined, you just let it pass

After quant_aware training -> 96% accuracy and 18 ms execution time
Much better accuracy but is it a fair comparison since you trained one epoch more?


### Triton
Python-like programming language to write optimized kernels



### Stable diffusion
Compiled unet: 
## kuber_docker
Orchestrate kubernetis locally via minikube

import torch
import torch_tensorrt
import torchvision.models as models

# We begin by defining and initializing a model
model = models.resnet18(pretrained=True).cuda().eval()

# Define sample inputs
inputs = torch.randn((16, 3, 224, 224)).cuda()
# Next, we compile the model using torch_tensorrt.compile
# We use the `ir="dynamo"` flag here, and `ir="torch_compile"` should
# work with cudagraphs as well.
opt = torch_tensorrt.compile(
    model,
    ir="dynamo",
    inputs=torch_tensorrt.Input(
        min_shape=(1, 3, 224, 224),
        opt_shape=(8, 3, 224, 224),
        max_shape=(16, 3, 224, 224),
        dtype=torch.float,
        name="x",
    ),
)

# We can enable the cudagraphs API with a context manager
with torch.no_grad():
    with torch_tensorrt.runtime.enable_cudagraphs(opt) as cudagraphs_module:
        out_trt = cudagraphs_module(inputs)

    # Alternatively, we can set the cudagraphs mode for the session
    torch_tensorrt.runtime.set_cudagraphs_mode(True)
    out_trt = opt(inputs)

    # We can also turn off cudagraphs mode and perform inference as normal
    torch_tensorrt.runtime.set_cudagraphs_mode(False)
    out_trt = opt(inputs)

    # If we provide new input shapes, cudagraphs will re-record the graph
inputs_2 = torch.randn((8, 3, 224, 224)).cuda()
inputs_3 = torch.randn((4, 3, 224, 224)).cuda()

with torch.no_grad():
    with torch_tensorrt.runtime.enable_cudagraphs(opt) as cudagraphs_module:
        out_trt_2 = cudagraphs_module(inputs_2)
        out_trt_3 = cudagraphs_module(inputs_3)


# wrapped runtime module with CUDA Graphs allows you
#  to encapsulate sequences of operations into graphs that can be executed efficiently,
#  even in the presence of graph break
class SampleModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu((x + 2) * 0.5)


model = SampleModel().cuda().eval()
input = torch.randn((1, 3, 224, 224)).cuda()

# The 'torch_executed_ops' compiler option is used in this example to intentionally introduce graph breaks within the module.
# Note: The Dynamo backend is required for the CUDA Graph context manager to handle modules in an Ahead-Of-Time (AOT) manner.
opt_with_graph_break = torch_tensorrt.compile(
    model,
    ir="dynamo",
    inputs=[input],
    min_block_size=1,
    pass_through_build_failures=True,
    torch_executed_ops={"torch.ops.aten.mul.Tensor"}, # this introduces a graph break
)

with torch.no_grad():
    with torch_tensorrt.runtime.enable_cudagraphs(
        opt_with_graph_break
    ) as cudagraphs_module:
        cudagraphs_module(input)

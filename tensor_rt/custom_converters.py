from typing import Dict, Sequence, Tuple, Union

import tensorrt as trt
import torch
import torch_tensorrt
from torch.fx.node import Argument, Target
from torch_tensorrt.dynamo.conversion import ConversionContext

class GeLU(torch.nn.Module):
    def __init__(self, mode="tanh"):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.gelu(x, approximate=self.mode)



@torch_tensorrt.dynamo.conversion.dynamo_tensorrt_converter(
    # The PyTorch operation to convert, when this operation is encountered, this converter will be called
    torch.ops.aten.gelu.default,
    # Validators are functions that determine that given a specific node, if it can be converted by the converter
    # otherwise they are run with pytorch
    # useful when you want to run custom validators only in specific cases
    capability_validator=lambda node, settings: (
        "approximate" in node.kwargs and node.kwargs["approximate"] == "tanh"
    ),
    # Can this converter be used in cases where the input shapes are dynamic
    supports_dynamic_shapes=True,
    # Set the priority of the converter to supersede the default one
    # you can have priority HIGH and STANDARD. HIGH is prepended and STANDARD is appended
    # first coverter in the list is used
    priority=torch_tensorrt.dynamo.conversion.ConverterPriority.HIGH,
    # Whether the converter requires a dynamic output allocator to run (e.g. data dependent ops)
    requires_output_allocator=True,
)

def aten_ops_gelu(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
    # The schema for torch.ops.aten.gelu.default is gelu(Tensor self, *, str approximate=’none’) -> Tensor

    from torch_tensorrt.dynamo import SourceIR

    # contains basic function to avoid re-implementations
    from torch_tensorrt.dynamo.conversion import impl

    # Cheap way to allow layer names to be unqiue
    op_count = 0

    def get_op_count():
        nonlocal op_count
        op_count += 1
        return op_count

    mul = lambda x, y: impl.elementwise.mul(
        ctx,
        target,
        name=f"mul_{get_op_count()}",
        source_ir=SourceIR.ATEN,
        lhs_val=x,
        rhs_val=y,
    )
    add = lambda x, y: impl.elementwise.add(
        ctx,
        target,
        name=f"add_{get_op_count()}",
        source_ir=SourceIR.ATEN,
        lhs_val=x,
        rhs_val=y,
    )
    tanh = lambda x: impl.activation.tanh(
        ctx, target, name=f"tanh_{get_op_count()}", source_ir=SourceIR.ATEN, input_val=x
    )

    # So we know that our custom converter is being run instead of the standard one
    print("\n\n---------------------------")
    print("Using custom GeLU converter")
    print("---------------------------\n\n")

    x_7 = mul(args[0], 0.5)
    x_8 = mul(args[0], 0.79788456080000003)
    x_9 = mul(args[0], 0.044714999999999998)
    x_10 = mul(x_9, args[0])
    x_11 = add(x_10, 1.0)
    x_12 = mul(x_8, x_11)
    x_13 = tanh(x_12)
    x_14 = add(x_13, 1.0)
    x_15 = mul(x_7, x_14)

    return x_15


my_mod = GeLU(mode="tanh").to("cuda").eval()
ex_input = torch.randn(2, 5).to("cuda")
my_standard_gelu = torch_tensorrt.compile(
    my_mod, arg_inputs=(ex_input,), min_block_size=1
)
print(my_standard_gelu.graph)
print(my_standard_gelu(ex_input))

my_custom_gelu = torch_tensorrt.compile(
    my_mod, arg_inputs=(ex_input,), min_block_size=1
)
with torch.no_grad():
    print(my_custom_gelu.graph)
    print(my_custom_gelu(ex_input))

# verify that in case of mode!=approximate the converter is not used
my_mod_erf = GeLU(mode="none").to("cuda").eval()
my_gelu_erf = torch_tensorrt.compile(
    my_mod_erf, arg_inputs=(ex_input,), min_block_size=1
)

with torch.no_grad():
    print(my_gelu_erf.graph)
    print(my_gelu_erf(ex_input))

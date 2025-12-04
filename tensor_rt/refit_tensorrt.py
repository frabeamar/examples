import numpy as np
import torch
import torch_tensorrt as torch_trt
import torchvision.models as models
from torch_tensorrt.dynamo import refit_module_weights


def refit():
    np.random.seed(0)
    torch.manual_seed(0)
    inputs = [torch.rand((1, 3, 224, 224)).to("cuda")]
    # a model with random weights

    model = models.resnet18(pretrained=False).to("cuda").eval()

    exp_program = torch.export.export(model, tuple(inputs))
    enabled_precisions = {torch.float}
    workspace_size = 20 << 30
    min_block_size = 0
    use_python_runtime = False
    torch_executed_ops = {}
    trt_gm = torch_trt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        min_block_size=min_block_size,
        torch_executed_ops=torch_executed_ops,
        immutable_weights=False,  # the compiled version should support weight refitting
        reuse_cached_engines=False,
        #
    )  # Output is a torch.fx.GraphModule

    # Save the graph module as an exported program
    torch_trt.save(trt_gm, "./compiled.ep", inputs=inputs)
    # Create and compile the updated model
    model2 = models.resnet18(pretrained=True).to("cuda").eval()
    exp_program2 = torch.export.export(model2, tuple(inputs))

    compiled_trt_ep = torch_trt.load("./compiled.ep")

    # This returns a new module with updated weights
    new_trt_gm = refit_module_weights(
        compiled_module=compiled_trt_ep,
        # name matching is complicated, that's why you pass
        # the compiled program to generate near identical names
        # this is however expensive
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        # to make weight name matching easier you can set this to true
        # tensorrt will save a map from pytorch to tensor rt names
        # this cache is stored as metadata
        # since the cache is based on a heuristic it might be necessary to verify the model
        # use_weight_map_cache = True,
        # verify_output = True
        ### if I set this two to True then I have a input type error on = exp_program2.module()(*inputs)
        ### one is on cuda and the other is not. exp_program is modified in place?
    )

    breakpoint()
    # Check the output
    with torch.no_grad():
        expected_outputs = exp_program2.module()(*inputs)
        refitted_outputs = new_trt_gm(*inputs)
        for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
            assert torch.allclose(expected_output, refitted_output, 1e-2, 1e-2), (
                "Refit Result is not correct. Refit failed"
            )

    print("Refit successfully!")


refit()

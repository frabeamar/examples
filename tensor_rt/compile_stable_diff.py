import torch
from diffusers import DiffusionPipeline
import torch_tensorrt
import time

from train import save_checkpoint

def timeit(model, prompt):
    for _ in range(10):
        model(prompt)

    start = time.perf_counter()
    for _ in range(10):
        result = model(prompt)
    end = time.perf_counter()
    return result, (end - start) / 10


model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda:0"

# Instantiate Stable Diffusion Pipeline with FP16 weights
pipe = DiffusionPipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16
)
pipe = pipe.to(device)
prompt="a castle in the sky"
imt, no_compile_ex_time = timeit(pipe, prompt)

# Optimize the UNet portion with Torch-TensorRT
pipe.unet = torch.compile(
    pipe.unet,
    backend="torch_tensorrt",
    options={
        "truncate_long_and_double": True,
        "enabled_precisions": {torch.float32, torch.float16},
    },
    dynamic=False,
)
save_checkpoint({"model_state_dict": pipe.unet.state_dict()}, "unet.pth")
with torch.no_grad():
    diffusion_output, compiled_ex_time = timeit(pipe, "a castle in the clouds")
    image = diffusion_output.images[0]
    image.save("majestic_castle.png")
    image.show()

print(f"Compiled: {compiled_ex_time} \t No compiled {no_compile_ex_time}")

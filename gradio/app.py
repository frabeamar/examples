import gradio as gr
import numpy as np
import torch
from diffusers import AutoPipelineForText2Image
from transformers import pipeline

pipe = AutoPipelineForText2Image.from_pretrained(
    "segmind/SSD-1B", torch_dtype=torch.float16
).to("cuda")

image = pipe("A futuristic city skyline at sunset").images[0]
breakpoint()
pipe("a photograph of an astronaut riding a horse").images[0].save(
    "astronaut_rides_horse.png"
)


print("running")
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda:0"
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")


def transcribe(stream, new_chunk):
    if new_chunk is None:
        return stream, ""
    sr, y = new_chunk
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y
    return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]


def predict(prompt):
    with torch.no_grad():
        img = pipe(prompt).images[0]
    return img


with gr.Blocks() as demo:
    with gr.Tab("Generate image"):
        gr.Interface(
            predict,
            inputs="text",
            outputs="image",
        )

    with gr.Tab("Trainscrive text"):
        gr.Interface(
            transcribe,
            ["state", gr.Audio(sources=["microphone"], streaming=True)],
            ["state", "text"],
            live=True,
            api_name="predict",
        )


iface = demo.launch(server_name="0.0.0.0", server_port=7868, share=False)

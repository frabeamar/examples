FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

WORKDIR /workspace
RUN pip install --no-cache-dir gradio
RUN pip install diffusers transformers numpy torchaudio
# downloads the model
# RUN python -c 'DiffusionPipeline.from_pretrained(  model_id, revision="", torch_dtype=torch.float16 )'
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]

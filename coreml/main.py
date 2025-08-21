import torch
import onnx
import coremltools as ct
from ultralytics import YOLO
import onnxruntime as ort

if __name__ == "__main__":
    # Download the YOLOv8 model from Hugging Face
    model = YOLO("yolov8n.pt")
    H = 224
    W = 224

    # this does not work: error: [mutex.cc : 452] RAW: Lock blocking 0x10cf1bef8   @
    
    model.export(format="tflite")

    breakpoint()
    model.export(format="onnx", dynamic=True)
    # Export PyTorch model to ONNX
    dummy_input = torch.randn(1, 3, H, W)
    # path = model.export(format="onnx")  # return path to exported model
    torch.onnx.export(
        model.model,
        dummy_input,
        "model.onnx",
        verbose=False,
        opset_version=19,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=["images"],
    )

    onnx_model = onnx.load("model.onnx")
    # Visualize the ONNX model using Netron: https://netron.app
    onnx.checker.check_model(onnx_model)

    ort_sess = ort.InferenceSession("model.onnx")
    outputs = ort_sess.run(None, {"images": dummy_input.numpy()})
    outs = model.model(dummy_input)
    # assert np.all(np.isclose(out[0].numpy() , outputs[0], rtol=1e-01))
    # H : image height, W: image width

    model.export(format="coreml")
    ts = torch.jit.trace(
        model.model.eval(), dummy_input, strict=False
    )  # TorchScript model

    mlmodel = ct.convert(
        source="pytorch",
        model=ts,
        inputs=[ct.ImageType(shape=(1, 3, H, W), scale=1 / 255, bias=[0, 0, 0])],
        classifier_config=None,
        convert_to="mlprogram",
    )

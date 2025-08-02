from collections import deque
from dataclasses import dataclass
import itertools
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from drawing import draw_prediction_on_image
import numpy as np
import cv2
import os
import time
from dataclasses import field
from threading import Thread, Event

from cropping import init_crop_region, determine_crop_region, run_inference

# Import matplotlib libraries
from matplotlib import pyplot as plt
# Some modules to display an animation using imageio.

model_name = "movenet_lightning"


def load_video(video_path):
    """Load a video file and return the frames."""
    cap = cv2.VideoCapture(video_path)
    for frames in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from video.")
            break
        yield frame


@dataclass
class ReadCameraInput:
    _cap: cv2.VideoCapture 
    _thread: Thread | None = None
    _stop_event: Event | None = None
    q: deque = field(default_factory=lambda: deque(maxlen=10))

    @classmethod
    def from_stream(cls, stream_url="udp://@:1234"):
        """Start reading frames from the UDP stream in a background thread."""
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError("Error: Cannot open UDP stream")
        return cls(cap)
    
    def start(self):
        stop_event = Event()
        # Create a deque to hold frames, with a maximum length of 10
        stop_event.clear()
        self._stop_event = stop_event 
        thread = Thread(target=self._update, daemon=True)
        self._thread = thread
        thread.start()
        return self

    def _update(self):
        """Background thread function to read frames and fill the deque."""
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                print("No frame received or stream ended.")
                break
            self.q.append(frame)

        self._cap.release()

    def stop(self):
        """Stop the background thread and release the video capture."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()

    def read(self):
        """Yield frames from the deque."""
        while True:
            if self.q:
                yield self.q.popleft()[..., ::-1]  # Convert BGR to RGB
            else:
                # No frames available, optionally wait a bit or break
                # You can add: time.sleep(0.01)
                continue




def download_model(model_name):
    if "tflite" in model_name:
        if "movenet_lightning_f16" in model_name:
            os.system(
                "wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"
            )
            input_size = 192
        elif "movenet_thunder_f16" in model_name:
            os.system(
                "wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite"
            )
            input_size = 256
        elif "movenet_lightning_int8" in model_name:
            os.system(
                "wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite"
            )
            input_size = 192
        elif "movenet_thunder_int8" in model_name:
            os.system(
                "wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite"
            )
            input_size = 256
        else:
            raise ValueError("Unsupported model name: %s" % model_name)

        # Initialize the TFLite interpreter
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()

        def movenet(input_image):
            """Runs detection on an input image.

            Args:
              input_image: A [1, height, width, 3] tensor represents the input image
                pixels. Note that the height/width should already be resized and match the
                expected input resolution of the model before passing into this function.

            Returns:
              A [1, 1, 17, 3] float numpy array representing the predicted keypoint
              coordinates and scores.
            """
            # TF Lite format expects tensor type of uint8.
            input_image = tf.cast(input_image, dtype=tf.uint8)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]["index"], input_image.numpy())
            # Invoke inference.
            interpreter.invoke()
            # Get the model prediction.
            keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])
            return keypoints_with_scores

    else:
        if "movenet_lightning" in model_name:
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
            input_size = 192
        elif "movenet_thunder" in model_name:
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
            input_size = 256
        else:
            raise ValueError("Unsupported model name: %s" % model_name)

        def movenet(input_image):
            """Runs detection on an input image.

            Args:
              input_image: A [1, height, width, 3] tensor represents the input image
                pixels. Note that the height/width should already be resized and match the
                expected input resolution of the model before passing into this function.

            Returns:
              A [1, 1, 17, 3] float numpy array representing the predicted keypoint
              coordinates and scores.
            """
            model = module.signatures["serving_default"]

            # SavedModel format expects tensor type of int32.
            input_image = tf.cast(input_image, dtype=tf.int32)
            # Run model inference.
            outputs = model(input_image)
            # Output is a [1, 1, 17, 3] tensor.
            keypoints_with_scores = outputs["output_0"].numpy()
            return keypoints_with_scores

    return movenet, input_size


def run_default(input_image, movenet, input_size):
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

    # Run model inference.
    keypoints_with_scores = movenet(input_image)

    # Visualize the predictions with image.
    display_image = tf.expand_dims(image, axis=0)
    display_image = tf.cast(
        tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32
    )
    output_overlay = draw_prediction_on_image(
        np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores
    )
    # plt.imshow(output_overlay)
    # plt.show(block=False)
    # _ = plt.axis("off")


def run_with_cropping_algo(image, movenet, input_size):
    # Load the input image.

    num_frames, image_height, image_width, _ = image.shape
    crop_region = init_crop_region(image_height, image_width)

    output_images = []
    plt.figure(figsize=(5, 5))

    # bar = display(progress(0, num_frames - 1), display_id=True)
    for frame_idx in range(num_frames):
        keypoints_with_scores = run_inference(
            movenet,
            image[frame_idx, :, :, :],
            crop_region,
            crop_size=[input_size, input_size],
        )
        output_image = draw_prediction_on_image(
            image[frame_idx, :, :, :].numpy().astype(np.int32),
            keypoints_with_scores,
            crop_region=None,
            close_figure=True,
            output_image_height=300,
        )
        crop_region = determine_crop_region(
            keypoints_with_scores, image_height, image_width
        )
        # bar.update(progress(frame_idx, num_frames - 1))
        plt.imshow(output_image)
        plt.savefig("output_image.png")


if __name__ == "__main__":
    movenet, input_size = download_model(model_name)
    path = Path.home() / ("data/output_h264.mp4")
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    plt.figure(figsize=(15, 15))

    # for image in load_video(path):
    input_stream = ReadCameraInput.from_stream().start()
    for image in input_stream.read():
        print("Processing frame...")
        # plt.imshow(image)
        # plt.axis("off")
        # plt.savefig("input_image.png", bbox_inches="tight", pad_inches=0.0)
        # transform numpy to tensor
        # input_image = tf.convert_to_tensor(image, dtype=tf.float32)
        input_image = tf.expand_dims(image, axis=0)
        time = time.time()
        run_default(input_image, movenet, input_size)
        print(time.time() - time)

        # run_with_cropping_algo(input_image, movenet, input_size)

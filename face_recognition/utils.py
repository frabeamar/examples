from dataclasses import dataclass
from pathlib import Path
from venv import logger

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from insightface.app import FaceAnalysis


def read_image(filename: str | Path) -> np.ndarray:
    logger.debug("Reading image: {}".format(filename))
    image = cv2.imread(str(filename), cv2.IMREAD_COLOR)  # pyright: ignore[reportCallIssue]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        raise IOError("Failed to load image: {}".format(filename))
    return image


class NoFaceDetected(Exception):
    pass


class WarpingException(Exception):
    pass


@dataclass
class Prediction:
    bbox: np.ndarray
    kps: np.ndarray
    det_score: np.float32
    landmark_3d_68: np.ndarray
    pose: np.ndarray
    landmark_2d_106: np.ndarray
    gender: np.int64
    age: int
    embedding: np.ndarray


class FaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = FaceAnalysis()
        self.embedder.prepare(ctx_id=0, det_size=(640, 640))

    def forward(self, x: np.ndarray) -> list[Prediction]:
        faces = self.embedder.get(x)
        if len(faces) == 0:
            raise NoFaceDetected()
        return [Prediction(**f) for f in faces]

    def draw_faces(self, x: np.ndarray) -> np.ndarray:
        """
        Draws bounding boxes and landmarks on the input image for detected faces.
        Args:
            x (np.ndarray): The input image (RGB format).
        """
        faces = self.embedder.get(x)
        rimg = self.embedder.draw_on(x, faces)
        return rimg


@dataclass
class PreprocessPipeline:
    detector: FaceModel = FaceModel()

    def __call__(
        self, filename: str | Path
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Processes a given image file by detecting faces, aligning them to a template,
        and cropping them.

        Args:
            filename (Union[str, Path]): The path to the image file to preprocess.
        """
        image = read_image(str(filename))
        detections: list[Prediction] = self.detector(image)
        if len(detections) > 1:
            logger.warning(
                f"Detected {len(detections)} faces in {filename}.\nSplitting up"
            )
        template, indices = self.create_template()

        aligned_images = []
        crops = []
        landmarks = []
        for i, detection in enumerate(detections):
            try:
                aligned = self.align(
                    image,
                    prediction=detection,
                    template=template,
                    indices=indices,
                    scale=1.0,
                )
            except WarpingException as e:
                logger.error(f"Could not align face {i} in {filename}: {e}")
            aligned_images.append(aligned)
            crop, landmark = self.crop(image, detection)
            crops.append(crop)
            landmarks.append(landmark)

        return aligned_images, crops, landmarks

    def display_detected(
        self,
        aligned: list[np.ndarray],
        crops: list[np.ndarray],
        landmarks: list[np.ndarray],
    ):
        """
        Displays and saves the detected, cropped, and aligned faces using matplotlib.
        """
        assert len(crops) == len(aligned) == len(landmarks), (
            "Input lists must have the same length."
        )
        plt.figure(figsize=(20, 10))
        fig, axs = plt.subplots(len(crops), 2)
        assert len(crops) == len(aligned) == len(landmarks)
        for i in range(len(aligned)):
            if i == 0:
                axs[i, 0].set_title("Original - cropped")
                axs[i, 1].set_title("Aligned")
            landmark = landmarks[i]
            axs[i, 0].imshow(crops[i])
            axs[i, 0].scatter(landmark[:, 0], landmark[:, 1], s=1)
            axs[i, 1].imshow(aligned[i])
            axs[i, 0].axis("off")
            axs[i, 1].axis("off")
        plt.tight_layout
        plt.savefig("aligned.png", dpi=150)
        plt.close()

    def create_template(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates a normalized landmark template from a reference

        Returns:
            - normalized_landmarks (np.ndarray): Normalized 2D landmarks [0, 1] of the reference face.
            - inner_eyes_nose_indices (np.ndarray): Indices of specific landmarks (inner eyes and nose) used for alignment.
        """
        img = read_image("front_portrait.jpg")
        [face] = self.detector(img)
        bbox = face.bbox
        top_left = np.array([bbox[0], bbox[1]])
        normalized_landmarks = (np.array(face.landmark_2d_106) - top_left) / np.array(
            [[bbox[2] - bbox[0], bbox[3] - bbox[1]]]
        )
        plt.figure(figsize=(10, 10))
        plt.scatter(normalized_landmarks[:, 0], 1 - normalized_landmarks[:, 1])
        for i, (x, y) in enumerate(normalized_landmarks):
            plt.text(x, 1 - y, str(i), color="red", fontsize=12)
        plt.savefig("landmark_template.png")
        inner_eyes_nose_indices = np.array([39, 81, 53])
        return normalized_landmarks, inner_eyes_nose_indices

    def get_all_bounding_boxes(self, image: np.ndarray)->np.ndarray:
        """
        Detects all faces in an image and returns their bounding boxes.
        Args:
            image (np.ndarray): The input image (RGB format).
        Returns:
            np.ndarray [N, 4]: An array of N bounding boxes, where each box is `[x1, y1, x2, y2]`.
        """
        faces = self.detector(image)
        return np.array([f.bbox for f in faces])

    def get_largest_bounding_box(self, predictions: list[Prediction]) -> np.ndarray:
        def area(bbox: np.ndarray) -> float:
            return (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])

        return sorted(
            [(i, p.bbox) for i, p in enumerate(predictions)],
            key=lambda x: area(x[1]),
        )[0][1]

    def crop(
        self, image: np.ndarray, prediction: Prediction
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Crops a face region from an image based on a given prediction's bounding box
        and adjusts the landmarks to be relative to the cropped region.

        Args:
            image (np.ndarray): The original image.
            prediction (Prediction): A `Prediction` object containing face details.

        Returns:
            - cropped_face (np.ndarray): The cropped face image.
            - relative_landmarks (np.ndarray): Landmarks adjusted to be relative to the top-left corner of the cropped face.
        """
        x, y, xx, yy = map(int, prediction.bbox)
        landmarks = prediction.landmark_2d_106 - np.array([x, y])
        return image[y:yy, x:xx], landmarks

    def align(
        self,
        image: np.ndarray,
        prediction: Prediction,
        template: np.ndarray,
        indices: np.ndarray,
        scale: float = 1,
    ) -> np.ndarray:
        """
        Aligns a detected face in an image to a predefined template using affine transformation.

        Args:
            image (np.ndarray): The original image.
            prediction (Prediction): A `Prediction` object for the face to align.
            template (np.ndarray): The normalized landmark template.
            indices (np.ndarray): Indices of landmarks to use for alignment.
            scale (float, optional): Scaling factor for the alignment. Defaults to 1.

        Returns:
            np.ndarray: The aligned face image.

        """
        face, landmarks = self.crop(image, prediction)
        crop_dim = np.array(face.shape[:2])
        tmp_min, tmp_max = np.min(template, axis=0), np.max(template, axis=0)
        min_max_tmp = (template - tmp_min) / (tmp_max - tmp_min)
        normalized_tmpl = (
            crop_dim * min_max_tmp[indices] * scale + crop_dim * (1 - scale) / 2
        )
        H = cv2.getAffineTransform(
            landmarks[indices].astype(np.float32),
            normalized_tmpl.astype(np.float32),
        )
        thumbnail = cv2.warpAffine(face, H, dsize=crop_dim)  # pyright: ignore
        if thumbnail is None:
            raise WarpingException("Could not align face")
        return thumbnail


def draw_bbox(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    image_with_boxes = image.copy()
    for box in boxes:
        x, y, width, height = box
        cv2.rectangle(
            image,
            (int(x), int(y)),
            (int(x + width), int(y + height)),
            color=(255, 0, 0),
            thickness=2,
        )
    plt.imshow(image)
    plt.axis("off")
    plt.savefig("detected_faces.png")
    return image_with_boxes


if __name__ == "__main__":
    pip = PreprocessPipeline()
    aligned, crops, landmarks = pip("group_picture.jpg")
    pip.display_detected(aligned, crops, landmarks)

from dataclasses import dataclass
from pathlib import Path
from venv import logger

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from insightface.app import FaceAnalysis


def read_image(filename):
    logger.debug("Reading image: {}".format(filename))
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError("Failed to load image: {}".format(filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

    def forward(self, x) -> list[Prediction] | None:
        faces = self.embedder.get(x)
        if len(faces) == 0:
            raise NoFaceDetected()
        return [Prediction(**f) for f in faces]

    def draw_faces(self, x):
        faces = self.embedder.get(x)
        rimg = self.embedder.draw_on(x, faces)
        return rimg


@dataclass
class PreprocessPipeline:
    detector: FaceModel = FaceModel()

    def __call__(self, filename: str | Path):
        image = read_image(str(filename))
        detections: list[Prediction] = self.detector(image)
        if len(detections) > 1:
            logger.warning(
                f"Detected {len(detections)} faces in {filename}.\nSplitting up"
            )
        template, indices = self.create_template()
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

            face, landmarks = self.crop(image, detection)
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(face)
            axs[1].imshow(aligned)
            plt.savefig(f"aligned_{Path(filename).stem}_{i}.png")
            plt.close()

    def create_template(self) -> tuple[np.ndarray, np.ndarray]:
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

    def get_all_bounding_boxes(self, image: np.ndarray):
        faces = self.detector(image)
        return np.array([f.bbox for f in faces])

    def get_largest_bounding_box(self, predictions: Prediction):
        def area(bbox: np.ndarray):
            return (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])

        return sorted(
            [(i, p.bbox) for i, p in enumerate(predictions)],
            key=lambda i, x: area(x),
            reverse=True,
        )[0][1]

    def crop(
        self, image: np.ndarray, prediction: Prediction
    ) -> [np.ndarray, np.ndarray]:
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
        thumbnail = cv2.warpAffine(face, H, dsize=crop_dim)
        if thumbnail is None:
            raise WarpingException("Could not align face")
        return thumbnail


def draw_bbox(image: np.ndarray, boxes: np.ndarray):
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
    return image


if __name__ == "__main__":
    pip = PreprocessPipeline()
    pip.create_template()
    aligned = pip("group_picture.jpg")

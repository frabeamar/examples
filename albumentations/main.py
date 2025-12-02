import albumentations as A
import cv2
import numpy as np


def augment_with_bboxes():
    train_transform = A.Compose(
        [
            A.RandomCrop(width=450, height=450, p=1.0),  # Example random crop
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="coco",  # Specify input format
            label_fields=["class_labels"],
            # specify them to ensure they are not dropped
        ),
        strict=True,
    )
    # safe bboxes augmentations
    A.AtLeastOneBBoxRandomCrop(max_attempts=10, p=0.5)
    A.BBoxSafeRandomCrop(width=300, height=300, p=0.5)  # all are visible
    A.RandomSizedBBoxSafeCrop(
        height=300, width=300, p=0.5
    )  # all are visible -> crops have random aspect ratios and are then resized

    augmented = train_transform(image=image, bboxes=bboxes, class_labels=class_labels)

    transformed_image = augmented["image"]
    transformed_bboxes = augmented["bboxes"]
    # Access transformed labels using the key from label_fields
    transformed_class_labels = augmented["class_labels"]


def augment_with_keypoints():
    transform = A.Compose(
        [
            A.RandomCrop(width=330, height=330),
            A.Affine(p=1, scale=0.8, shear=5, translate_percent=0.05, rotate=15),
            A.RandomBrightnessContrast(p=0.2),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
    )
    class_labels = np.array(
        [
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "right_hip",
        ]
    )


def semantic_segmentation_augmentation():
    TARGET_SIZE = (256, 256)  # Example input height, width

    train_transform = A.Compose(
        [
            # Resize shortest side of image and mask to TARGET_SIZE * 2, maintaining aspect ratio
            A.SmallestMaxSize(max_size=TARGET_SIZE[0] * 2, p=1.0),
            A.RandomCrop(height=TARGET_SIZE[0], width=TARGET_SIZE[1], p=1.0),
            A.Rotate(limit=15, p=0.5, interpolation=cv2.INTER_NEAREST),
            # Apply one of 8 random symmetries (flips/rotations)
            A.SquareSymmetry(p=1.0),  # Replaces Horizontal/Vertical Flips
            # Optional: Further rotation if SquareSymmetry isn't sufficient or desired
            # A.Rotate(limit=30, p=0.3),
            # Optional: Image-only transforms (applied only to image)
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(std_range=(0.1, 0.2), p=0.2),
            # --- Framework-specific steps ---
            # Normalize image (mask is not normalized)
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # Convert image and mask to PyTorch tensors
            A.ToTensorV2(),
        ]
    )


def strong_augmentation():
    strong_train_transforms = A.Compose(
        [
            A.RandomResizedCrop(224, 224, scale=(0.6, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2
                    ),
                    A.ToGray(p=1.0),
                ],
                p=0.8,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(1, 3)),
                    A.GaussNoise(var_limit=(10, 50)),
                ],
                p=0.5,
            ),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ToTensorV2(),
        ]
    )

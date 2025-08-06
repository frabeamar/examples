# @title Helper functions for visualization
from tensorflow_docs.vis import embed
import numpy as np
import cv2

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
from utils import KEYPOINT_EDGE_INDS_TO_COLOR

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

coco_to_mp_map = [
    0,  # 0: Nose
    1,  # 1: Left Eye
    2,  # 2: Right Eye
    3,  # 3: Left Ear
    4,  # 4: Right Ear
    11,  # 5: Left Shoulder
    12,  # 6: Right Shoulder
    13,  # 7: Left Elbow
    14,  # 8: Right Elbow
    15,  # 9: Left Wrist
    16,  # 10: Right Wrist
    23,  # 11: Left Hip
    24,  # 12: Right Hip
    25,  # 13: Left Knee
    26,  # 14: Right Knee
    27,  # 15: Left Ankle
    28,  # 16: Right Ankle
]


def draw_landmarks_on_image(rgb_image, keypoints_with_scores):
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    # Draw the pose landmarks.
    pose_landmarks = np.zeros((1, 1, 33, 3), dtype=np.float32)
    pose_landmarks[:, :, coco_to_mp_map]=keypoints_with_scores
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend(
        [
            landmark_pb2.NormalizedLandmark(x=x, y=y, z=0)
            for x, y in keypoints_with_scores[0, 0, :, :2]
        ]
    )
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style(),
    )

    return annotated_image


def _keypoints_and_edges_for_display(
    keypoints_with_scores, height, width, keypoint_threshold=0.11
):
    """Returns high confidence keypoints and edges for visualization.

    Args:
      keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
      height: height of the image in pixels.
      width: width of the image in pixels.
      keypoint_threshold: minimum confidence score for a keypoint to be
        visualized.

    Returns:
      A (keypoints_xy, edges_xy, edge_colors) containing:
        * the coordinates of all keypoints of all detected entities;
        * the coordinates of all skeleton edges of all detected entities;
        * the colors in which the edges should be plotted.
    """
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1
        )
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :
        ]
        keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (
                kpts_scores[edge_pair[0]] > keypoint_threshold
                and kpts_scores[edge_pair[1]] > keypoint_threshold
            ):
                x_start = kpts_absolute_xy[edge_pair[0], 0]
                y_start = kpts_absolute_xy[edge_pair[0], 1]
                x_end = kpts_absolute_xy[edge_pair[1], 0]
                y_end = kpts_absolute_xy[edge_pair[1], 1]
                line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)
    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
    image,
    keypoints_with_scores,
    crop_region=None,
    close_figure=False,
    output_image_height=None,
):
    """Draws the keypoint predictions on image.

    Args:
      image: A numpy array with shape [height, width, channel] representing the
        pixel values of the input image.
      keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
      crop_region: A dictionary that defines the coordinates of the bounding box
        of the crop region in normalized coordinates (see the init_crop_region
        function below for more detail). If provided, this function will also
        draw the bounding box on the image.
      output_image_height: An integer indicating the height of the output image.
        Note that the image aspect ratio will be the same as the input image.

    Returns:
      A numpy array with shape [out_height, out_width, channel] representing the
      image overlaid with keypoint predictions.
    """
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    # To remove the huge white borders
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis("off")

    im = ax.imshow(image)
    line_segments = LineCollection([], linewidths=(4), linestyle="solid")
    ax.add_collection(line_segments)
    # Turn off tick labels
    scat = ax.scatter([], [], s=60, color="#FF1493", zorder=3)

    (keypoint_locs, keypoint_edges, edge_colors) = _keypoints_and_edges_for_display(
        keypoints_with_scores, height, width
    )

    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
    if keypoint_edges.shape[0]:
        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)
    if keypoint_locs.shape[0]:
        scat.set_offsets(keypoint_locs)

    if crop_region is not None:
        xmin = max(crop_region["x_min"] * width, 0.0)
        ymin = max(crop_region["y_min"] * height, 0.0)
        rec_width = min(crop_region["x_max"], 0.99) * width - xmin
        rec_height = min(crop_region["y_max"], 0.99) * height - ymin
        rect = patches.Rectangle(
            (xmin, ymin),
            rec_width,
            rec_height,
            linewidth=1,
            edgecolor="b",
            facecolor="none",
        )
        ax.add_patch(rect)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)

    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )[..., 1:]
    plt.close(fig)
    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image_from_plot = cv2.resize(
            image_from_plot,
            dsize=(output_image_width, output_image_height),
            interpolation=cv2.INTER_CUBIC,
        )
    return image_from_plot


def to_gif(images, duration):
    """Converts image sequence (4D numpy array) to gif."""
    imageio.mimsave("./animation.gif", images, duration=duration)
    return embed.embed_file("./animation.gif")


def progress(value, max=100):
    return HTML(
        """
      <progress
          value='{value}'
          max='{max}',
          style='width: 100%'
      >
          {value}
      </progress>
  """.format(value=value, max=max)
    )

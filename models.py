from kornia.io import load_image
from kornia.feature import DeDoDe, DISK, match_mnn, LightGlueMatcher, LoFTR
from pathlib import Path
from enum import Enum
import torch
from transformers import DetrForObjectDetection
from torchvision.io.image import decode_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
class ModelTypes(Enum):
    DETR = "detr"


from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights

def get_torchvision_model():
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    preprocess = weights.transforms()
    return model, preprocess


def eval_with_torchvision_model(model:torch.nn.Module, weights: FasterRCNN_ResNet50_FPN_V2_Weights, batch:torch.Tensor):

    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                            labels=labels,
                            colors="red",
                            width=4, font_size=30)
    im = to_pil_image(box.detach())
    im.show()


def get_model():
        dedode = DeDoDe.from_pretrained(detector_weights="L-C4-v2", descriptor_weights="B-upright")
        images = torch.randn(1, 3, 256, 256)
        keypoints, scores, features = dedode(images) # alternatively do both



        disk = DISK.from_pretrained('depth')
        images = torch.rand(1, 3, 256, 256)
        features = disk(images)


        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        #Matching
        img1 = torch.rand(1, 1, 320, 200)
        img2 = torch.rand(1, 1, 128, 128)
        input = {"image0": img1, "image1": img2}
        LightGlueMatcher("superpoint")


        img1 = torch.rand(1, 1, 320, 200)
        img2 = torch.rand(1, 1, 128, 128)
        input = {"image0": img1, "image1": img2}
        loftr = LoFTR('outdoor')
        out = loftr(input)


get_model()

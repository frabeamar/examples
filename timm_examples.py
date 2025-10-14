"""
Comparison: Pooled vs Unpooled Features
---------------------------------------

Unpooled Features:
- Keep all feature activations at full resolution.
- Preserve fine-grained spatial/structural details 
    => Better for localization tasks (e.g., segmentation, detection, optical flow).
- More flexible: later layers can learn how to aggregate, instead of forcing an early summary.
- Disadvantage: 
    => high memory/computation cost.
    => less invariance to small shifts/variations.

Pooled Features:
- Apply aggregation (max, mean, attention, etc.) to reduce feature resolution.
- Create compact representations → faster training & lower memory use.
- Provide invariance to small spatial/nodal variations.
- Capture global context more easily.
- Disadvantage: loss of fine detail, risk of missing important local cues.
- Disadvantage: aggressive pooling may create an information bottleneck.

Rule of Thumb:
- Use unpooled features when you care about *where* things are.
- Use pooled features when you care about *what* things are.
"""

import torch
import timm
m = timm.create_model('xception41', pretrained=True)
# POOLED FEATURES
o = m(torch.randn(2, 3, 299, 299))
# UNPOOLED FEATURES
o = m.forward_features(torch.randn(2, 3, 299, 299))
print(f'Unpooled shape: {o.shape}')

# NO CLASSIFIER AND POOLING
m = timm.create_model('resnet50', pretrained=True, num_classes=0, global_pool='')
o = m(torch.randn(2, 3, 224, 224))

# REMOVE CLASSIFIER
m.reset_classifier(0, '')

# The last hidden state can be fed back into the head of the model
#  using the forward_head() function.
model = timm.create_model('vit_medium_patch16_reg1_gap_256', pretrained=True)
output = model.forward_features(torch.randn(2,3,256,256))
print('Unpooled output shape:', output.shape)
classified = model.forward_head(output)


# consistent interface for creating any of the included models 
# as feature backbones that output feature maps for selected levels.
m = timm.create_model('resnest26d', features_only=True, pretrained=True)
# query about the feature channel in each level
print(f'Feature channels: {m.feature_info.channels()}')
# look how much the input resolution was reduced
print(f'Feature reduction: {m.feature_info.reduction()}')
# Output stride (feature map dilation)
# out_indices -> only output sepecific feature maps, can be negative (starts from the back)
m = timm.create_model('ecaresnet101d', features_only=True, output_stride=8, out_indices=(2, 4), pretrained=True)


model = timm.create_model('vit_medium_patch16_reg1_gap_256', pretrained=True)
# extract the intermediate features of a vit
output, intermediates = model.forward_intermediates(torch.randn(2,3,256,256))
# can also remove intermediate layes
indices = model.prune_intermediate_layers(indices=(-2,), prune_head=True, prune_norm=True)  # prune head, norm, last block

#list models
timm.list_models(pretrained=True)
#list models with mattern
model_names = timm.list_models('*resne*t*')
# create a standard transform [resize, center_crop, to_tensor, normalize]
timm.data.create_transform((3, 224, 224))

# see with what transformation the model was trained with
model.pretrained_cfg
#filter only with the data config
data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
# can create the relative config
transform = timm.data.create_transform(**data_cfg)

# get the imagenet labels
IMAGENET_1k_URL = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'
IMAGENET_1k_LABELS = requests.get(IMAGENET_1k_URL).text.strip().split('\n')

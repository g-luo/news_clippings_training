import torch
from mmf.common.registry import registry
from mmf.models import BaseModel
from typing import Optional, Dict
from torch import nn, Tensor
from mmf.modules.layers import MLPClassifer
import clip
from news_clippings_training.models.utils import get_optimizer_parameters_custom, freeze_optimizer_parameters_clip
from collections import OrderedDict

EMBEDDING_DIMS = {"RN101": 512, "ViT-B/32": 512, "ViT-B/16": 512, "RN50": 1024, "RN50x16": 768}
REMAPPING = {"vitb32": "ViT-B/32", "rn101": "RN101", "rn50": "RN50"}

# Ensure backwards compatibility with NewsCLIPpings
@registry.register_model("clip")
@registry.register_model("clip_concat")
class CLIP(BaseModel):

  def __init__(self, config, *args, **kwargs):
    super().__init__(config, *args, **kwargs)
    clip_model_type = self.config.get("clip_model_type")
    clip_model_type = REMAPPING.get(clip_model_type, clip_model_type)
    self.clip_model_type = clip_model_type

  def build(self):
    self.model, _ = clip.load(self.clip_model_type, device="cuda", jit=False)
    print(f"Using CLIP model type {self.clip_model_type}")

    # Do this since original model is fp16
    self.model = self.model.float()
    self.model.config = self.config
    
    freeze_optimizer_parameters_clip(self.config, self.model)
    
    mlp_input_dim = self.config.get("mlp_input_dim", self.get_mlp_input_dim())
    mlp_hidden_dim = self.config.get("mlp_hidden_dim", EMBEDDING_DIMS[self.clip_model_type])
    self.model.classifier = nn.Sequential(
        MLPClassifer (
            mlp_input_dim,
            self.config.num_labels,
            hidden_dim=mlp_hidden_dim,
            num_layers=self.config.mlp_num_layers,
            dropout=self.config.mlp_dropout,
            hidden_act=self.config.mlp_act,
            batch_norm=False
        ),
        nn.Softmax(dim=1)
    )
    self.linear_probe = self.config.get("linear_probe", False)
  
  def get_mlp_input_dim(self):
    return EMBEDDING_DIMS[self.clip_model_type] * 2

  def get_optimizer_parameters(self, config):
    return get_optimizer_parameters_custom(config, self.model)

  def get_features(self, sample_list):
    features = []

    # Text features
    input_ids = sample_list["input_ids"]
    if self.config.get("image_only", False):
        input_ids = torch.ones(input_ids.shape, device="cuda").long()

    if self.linear_probe:
      with torch.no_grad():
        text_feature = self.model.encode_text(input_ids)
    else:
      text_feature = self.model.encode_text(input_ids)

    text_feature = text_feature.float()
    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
    text_feature.to("cuda")

    # Image features
    image = sample_list["image"]  
    if self.config.get("text_only", False):
        image = torch.ones(image.shape, device="cuda").long()

    if self.linear_probe:
      with torch.no_grad():
        image_feature = self.model.encode_image(image)
    else:
      image_feature = self.model.encode_image(image)
    
    image_feature = image_feature.float()
    image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
    image_feature.to("cuda")

    return text_feature, image_feature

  def process_features(self, text_feature, image_feature):
    return torch.cat([text_feature, image_feature], dim=-1)

  def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
    output_dict = {}

    text_feature, image_feature = self.get_features(sample_list)
    features = self.process_features(text_feature, image_feature)
    logits = self.model.classifier(features)
    reshaped_logits = logits.contiguous().view(-1, self.config.num_labels)

    output_dict["scores"] = reshaped_logits 
    return output_dict

@registry.register_model("clip_concat_dot")
class CLIPConcatDot(CLIP):

  def build(self):
    super().build()

  def get_mlp_input_dim(self):
    return EMBEDDING_DIMS[self.clip_model_type] * 2 + 1

  def process_features(self, text_feature, image_feature):
    dot_product = torch.sum(text_feature * image_feature, dim=-1).unsqueeze(1)
    return torch.cat([dot_product, text_feature, image_feature], dim=-1)
    
@registry.register_model("clip_mult")
class CLIPMult(CLIP):

  def build(self):
    super().build()

  def get_mlp_input_dim(self):
    return EMBEDDING_DIMS[self.clip_model_type]

  def process_features(self, text_feature, image_feature):
    return text_feature * image_feature
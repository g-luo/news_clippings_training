import torch
import json, pickle
import re
import numpy as np
from typing import Dict
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.visualize import visualize_images
from foil_mmf.foil.datasets.database import FOILAnnotationDatabase, FOILFeaturesDatabase
from foil_mmf.foil.utils import *

class FOILDataset(MMFDataset):
	def __init__(self, config: Dict, dataset_type: str, index, *args, **kwargs):
		super().__init__("foil", config, dataset_type, index, *args, **kwargs)
		np.random.seed(2)

	# Example annotations formatting
	# {"annotations": [sample_1, sample_2, ...]}
	def build_annotation_db(self):
		"""
		Override build_image_db and build_feature_db for to implement custom versions of
		those for a particular dataset.
		"""
		annotation_path = self._get_path_based_on_index(
			self.config, "annotations", self._index
		)
		return FOILAnnotationDatabase(self.config, annotation_path)

	def init_processors(self):
		super().init_processors()
		if self._use_images and hasattr(self, "image_processor"):
			self.image_db.transform = self.image_processor

	# def get_image_path(self, image_id: str):
	# 	"""
	# 	Utility function to convert COCO image id to actual image path.
	# 	Please note that only jpg images are currently supported.
	# 	"""
	# 	if self.config.dataset == "coco":
	# 		if self.dataset_type == "train":
	# 			image_path = f"COCO_train2014_{str(image_id).zfill(12)}.jpg"
	# 		elif self.dataset_type == "val":
	# 			image_path = f"COCO_train2014_{str(image_id).zfill(12)}.jpg"
	# 		else:
	# 			image_path = f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
	# 	else:
	# 		image_path = f"{str(image_id)}.jpg"
	# 	return image_path

	# def get_feature_path(self, image_path):
	# 	if "news" in self.config.dataset:
	# 		feature_path = image_path.replace(".jpg", ".npy").replace("images", "features")
	# 	else:
	# 		feature_path = image_path.replace(".jpg", ".npy")
	# 	return feature_path

	def __getitem__(self, idx: int):
		# ============================================
		# 			   Tokenize caption
		# ============================================
		sample_info = self.annotation_db[idx]
		return self.create_item(sample_info, idx)

	def create_item(self, sample_info, idx):
		# Example sample formatting
		# {
		#    id: 893
		#    caption: "this is a caption"
		#    foil: True
		# }
		current_sample = Sample()

		processed_caption = self.text_processor({"text": sample_info["caption"]})
		current_sample.update(processed_caption)
		current_sample.caption = sample_info["caption"]

		if "foil_id" in sample_info:
			foil_id = sample_info["foil_id"]
		elif "id" in sample_info:
			foil_id = sample_info["id"]
		else:
			foil_id = idx
			
		current_sample.id = torch.tensor(
			int(foil_id), dtype=torch.int
		)

		# ============================================
		# 			Add features / image
		# ============================================

		# if self._use_features or self._use_images:
		# 	current_sample["image_path"] = sample_info.get("image_path", self.get_image_path(sample_info.get("image_id", -1)))

		# if self._use_features:
		# 	sample_info["feature_path"] = sample_info.get("feature_path", self.get_feature_path(current_sample["image_path"]))
		# 	features = self.features_db[idx]
		# 	current_sample.update(features)
		
		# if self._use_images:
		# 	current_sample.image = self.image_db.from_path(current_sample["image_path"])["images"][0]
		
		# Foil and target word is unique to the FOIL task but can be empty
		current_sample.foil_word = sample_info.get("foil_word", "")
		current_sample.target_word = sample_info.get("target_word", "")

		target = sample_info.get("foil", current_sample.foil_word != "ORIG")
		current_sample.targets = torch.tensor(int(target), dtype=torch.long)
		
		return current_sample

	# ================================================
	#         Formatting used by mmf_predict
	# ================================================
	def format_for_prediction(self, report):
		output = []
		for idx, id in enumerate(report.id):
			target = report.targets[idx].item()
			# For binary classification this would result in 0 or 1
			score = report.scores[idx].argmax().item()
			caption = report.caption[idx]
			output.append({
											"id": id.item(), 
											"target": target, 
											"score": score,
											"caption": caption
										})
		return output

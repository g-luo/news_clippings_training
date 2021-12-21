from mmf.datasets.processors.processors import BaseProcessor
from mmf.common.registry import registry
import torch
import clip

@registry.register_processor("clip_tokenizer")
class CLIPTokenizer(BaseProcessor):
  def __init__(self, config, *args, **kwargs):
    super().__init__(config, *args, **kwargs)
    self.max_seq_length = config.get("max_seq_length", 77)

  def __call__(self, item):
    processed_item = {}
    input_ids = clip.tokenize([item["text"]], truncate=True)
    input_padding = torch.zeros(self.max_seq_length - input_ids.shape[1])

    processed_item["input_ids"] = torch.cat((input_ids, input_padding)).long()
    processed_item["input_ids"] = processed_item["input_ids"].squeeze(0)
    processed_item["lm_label_ids"] = -1 * torch.ones((self.max_seq_length, 1))
    processed_item["segment_ids"] = torch.zeros((self.max_seq_length, 1))
    processed_item["input_mask"] = processed_item["input_ids"] != 0
    return processed_item
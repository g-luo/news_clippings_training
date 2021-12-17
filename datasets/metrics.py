import torch
from mmf.common.registry import registry
from mmf.modules.metrics import BaseMetric

# https://github.com/facebookresearch/mmf/blob/357ee45986f542871e4778342331ed9e2594e174/website/docs/tutorials/metrics.md
@registry.register_metric("split_accuracy")
class SplitAccuracy(BaseMetric):
  """Metric for accuracy by class.
  """

  def __init__(self, split_key, split_names, target_names, target_breakdown=False):
    super().__init__("split_accuracy")
    self.split_names = split_names
    self.split_key = split_key
    self.target_names = target_names
    # Add this so tensor fields appropriately accumulate
    
    self.required_params = self.required_params + [self.split_key]
    self.target_breakdown = target_breakdown

  def calculate(self, sample_list, model_output, *args, **kwargs):
    output = model_output["scores"]
    expected = sample_list["targets"]
    if len(self.split_names) > 0:
      split_idxs = sample_list[self.split_key]

    assert (
      output.dim() <= 2
    ), "Output from model shouldn't have more than dim 2 for accuracy"
    assert (
      expected.dim() <= 2
    ), "Expected target shouldn't have more than dim 2 for accuracy"

    if output.dim() == 2:
      output = torch.max(output, 1)[1]

    if expected.dim() == 2 and expected.size(-1) != 1:
      expected = torch.max(expected, 1)[1]

    output = output.squeeze()
    result = {}
    for split_idx in range(len(self.split_names)):
      total = (split_idxs == split_idx)
      correct = torch.logical_and(expected == output, total)
      result[f"{self.split_names[split_idx]}_accuracy"] = correct.sum().float() / total.sum().float()
      if self.target_breakdown:
        for target_idx in range(len(self.target_names)):
          target_correct = torch.logical_and(correct, (expected == target_idx))
          target_total = torch.logical_and(total, (expected == target_idx))
          result[f"{self.split_names[split_idx]}_{self.target_names[target_idx]}_accuracy"] = target_correct.sum().float() / target_total.sum().float()
      
    for target_idx in range(len(self.target_names)):
      total = (expected == target_idx)
      correct = torch.logical_and(expected == output, total)
      result[f"{self.target_names[target_idx]}_accuracy"] = correct.sum().float() / total.sum().float()

    return result
 
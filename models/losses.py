from mmf.common.registry import registry
import torch.nn as nn
import torch

@registry.register_loss("cross_entropy_weighted")
class CrossEntropyLossWeighted(nn.Module):
  def __init__(self, **params):
    super().__init__()
    params["reduction"] = 'none'
    self.loss_fn = nn.CrossEntropyLoss(**params)

    print("Using cross entropy weighted loss.")

  def forward(self, sample_list, model_output):
    """
      Weight loss according to batch_weights and take the mean.
    """
    batch_weights = sample_list["batch_weights"]
    loss = self.loss_fn(model_output["scores"], sample_list["targets"])
    loss = loss * batch_weights
    loss = torch.mean(loss)
    return loss
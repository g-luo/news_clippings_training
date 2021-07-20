from mmf.common.registry import registry
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder
from foil_mmf.foil.datasets.foil import FOILDataset

@registry.register_builder("foil")
class FOILBuilder(MMFDatasetBuilder):
		def __init__(
			self, 
			dataset_name="foil", 
			dataset_class=FOILDataset,
			*args, 
			**kwargs,
		):
			super().__init__(dataset_name, dataset_class, *args, **kwargs)
	
		@classmethod
		def config_path(cls):
			return "configs/experiments/datasets/foil.yaml"
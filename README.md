# NewsCLIPpings Training Code

## Getting Started
To get started, install MMF. Here we provide suggested versions of MMF, torch, and torchvision that are known to be compatible with CLIP.
```
  pip install git+https://github.com/facebookresearch/mmf.git@08f062ef8cc605eed4a5dba729899c1cfc88a23b
  pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

## Data Setup
Download the dataset as instructed by our [dataset repo](https://github.com/g-luo/news_clippings#data-format). Also download the VisualNews dataset for the captions and images.

## Configs
Note that paths of files in the `configs/` folder such as `dataset_config.news_clippings.annotations` and `dataset_config.news_clippings.images` should be modified according to your machine.

## Training 
TODO

## Inference
Run inference in the main `news_clippings_training/` folder. Make sure `checkpoint.resume_pretrained` is set to False otherwise MMF will try to continue training your model during inference. To output predictions by sample run `mmf_predict`, else just output the accuracy in the `.out` file by replacing the command with `mmf_run`.

```
export resource_path="<path to our finetuned models>"
export model="clip"
export dataset="semantics_clip_text_image"
MMF_USER_DIR="." \
  nohup mmf_predict config="${resource_path}/${model}/${dataset}_${model}/config.yaml" \
  model=clip \
  dataset=news_clippings \
  run_type=test \
  checkpoint.resume_file="${resource_path}/${model}/${dataset}_${model}/test.ckpt" \
  env.save_dir=predictions/${dataset}/${dataset}_${model} \
  checkpoint.resume_pretrained=False \
  > ${dataset}_${model}.out &
```

## TODO
- [ x ] Add base configs
- [ x ] Add model code
- [ x ] Update dataset code
- [ x ] Add README with info about file location, annotation structure
- [ ] Add paper configs / weights
- [ ] Add script for populating annotations and image paths based on VisualNews



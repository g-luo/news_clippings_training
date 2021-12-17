#!/bin/bash
MMF_USER_DIR=$PWD \
  nohup mmf_predict config=configs/experiments/news_clippings.yaml \
  model=clip \
  dataset=news_clippings \
  run_type=test \
  checkpoint.resume_file=runs/${split}_${model}/best.ckpt \
  checkpoint.resume_pretrained=False \
  env.save_dir=runs/${split}_${model} \
  > ${split}_${model}_test.out &

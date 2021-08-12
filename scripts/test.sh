#!/bin/bash
MMF_USER_DIR="." \
  nohup mmf_predict config=${resource_path}/${experiment}/config.yaml \
  model=clip \
  dataset=news_clippings \
  run_type=test \
  checkpoint.resume_file=${resource_path}/${experiment}/test.ckpt \
  checkpoint.resume_pretrained=False \
  env.save_dir=runs/${split}_${model} \
  env.tensorboard_logdir=runs/${split}_${model} \
  > ${split}_${model}_test.out &

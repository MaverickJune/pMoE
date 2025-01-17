#!/bin/bash

docker run --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=1,3 \
  -it pmoe_wonjun
  # -v /home/wjbang/workspace/docker_repo/zfp:/workspace/zfp \
  # -v /home/wjbang/workspace/docker_repo/ScheMoe_Custom:/workspace/ScheMoe_Custom \
  
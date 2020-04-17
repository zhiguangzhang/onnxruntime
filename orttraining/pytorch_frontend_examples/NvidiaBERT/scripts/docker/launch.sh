#!/bin/bash

docker run -it --rm \
  --runtime=nvidia \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v ${PWD}:/workspace/bert \
  -v /bert_ort/wechi/repos/DeepLearningExamples/PyTorch/LanguageModeling/BERT/data/:/data \
  -v pytorch_frontend:/python/frontend \
  -v /bert_ort/chenta/bert_models:/bert_models \
  bert_pyt bash

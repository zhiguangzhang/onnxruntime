#!/bin/bash
docker build . --rm -t bert_pyt --build-arg ORT_WHEEL_NAME=onnxruntime_gpu-0.5.0-cp36-cp36m-linux_x86_64.whl

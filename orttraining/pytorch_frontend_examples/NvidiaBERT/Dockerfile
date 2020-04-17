# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#build based on azure ubuntu 16.04 image
ARG FROM_IMAGE_NAME=mcr.microsoft.com/azureml/bert:pretrain-openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04
ARG ORT_WHEEL_NAME=onnxruntime_gpu-0.5.0-cp36-cp36m-linux_x86_64.whl
FROM ${FROM_IMAGE_NAME}
RUN apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract openssh-client openssh-server aptitude

ADD upgrade_torch.sh /ort_script/

ENV LD_LIBRARY_PATH /opt/miniconda/envs/amlbert/lib:$LD_LIBRARY_PATH

RUN /bin/bash /ort_script/upgrade_torch.sh

ENV BERT_PREP_WORKING_DIR /workspace/bert/data

WORKDIR /workspace
RUN git clone https://github.com/attardi/wikiextractor.git
RUN git clone https://github.com/soskek/bookcorpus.git

RUN git clone https://github.com/NVIDIA/apex && cd apex \
    git checkout 3ae89c754d945e407a6674aa2006d5a0e35d540e && \
    /opt/miniconda/envs/amlbert/bin/python -m pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
 

ADD onnxruntime_gpu-0.5.0-cp36-cp36m-linux_x86_64.whl /ort_wheel/
ENV PYTHONPATH=/python

#patch the pytorch
ADD pytorch_patch/* /opt/miniconda/envs/amlbert/lib/python3.6/site-packages/torch/onnx/

WORKDIR /workspace/bert
RUN /opt/miniconda/envs/amlbert/bin/python -m pip install tqdm boto3 requests six ipdb h5py html2text nltk progressbar tensorboard==1.14 onnx mpi4py
RUN /opt/miniconda/envs/amlbert/bin/python -m pip install /ort_wheel/onnxruntime_gpu-0.5.0-cp36-cp36m-linux_x86_64.whl
COPY . .

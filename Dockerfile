FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /projects/ai_compiler_study

RUN apt-get update
RUN python -m pip install --upgrade pip
RUN pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

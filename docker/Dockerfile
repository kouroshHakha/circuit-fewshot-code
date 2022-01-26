FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

# install miniconda (comes with python 3.7 default)
ARG CONDA_VERSION
ARG CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh
RUN cd /stage && curl -fSsL --insecure ${CONDA_URL} -o install-conda.sh &&\
    /bin/bash ./install-conda.sh -b -p /opt/conda &&\
    /opt/conda/bin/conda clean -ya
ENV PATH=/opt/conda/bin:${PATH}


ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
RUN pip install pytorch-lightning==1.3.7
RUN pip install wandb plotly pyyaml IPython jinja2 h5py
RUN conda install ruamel.yaml

# wandb setup
ARG WANDB_API_KEY
ENV WANDB_API_KEY $WANDB_API_KEY

ARG GID
ARG UID
ARG UNAME
RUN sudo groupadd -g ${GID} -o ${UNAME}
RUN sudo useradd -m -u ${UID} -g $GID -o -s /bin/bash ${UNAME}

USER $UNAME

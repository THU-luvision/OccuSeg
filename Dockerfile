FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libnccl2=2.0.5-3+cuda9.0 \
         libnccl-dev=2.0.5-3+cuda9.0 \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*


ENV PYTHON_VERSION=3.6
RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \     
     rm ~/miniconda.sh && /opt/conda/bin/conda clean -ya
#     /opt/conda/bin/conda install conda-build && \

WORKDIR /workspace
RUN chmod -R a+w /workspace
COPY p1.yml .

RUN  /opt/conda/bin/conda env create -f p1.yml && \
     /opt/conda/bin/conda clean -ya 
ENV PATH /opt/conda/envs/p1/bin:$PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/workspace/extra/cudpp/build/lib:/workspace/extra/easy_profiler/build/bin

COPY . .
# Manually run following line in the container
# RUN bash all_build.sh
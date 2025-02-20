FROM nvcr.io/nvidia/pytorch:24.10-py3

# to avoid interaction with apt-get
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
	gengetopt \
	re2c \
	libomp-dev \
    	libsleef-dev \
	libgraphviz-dev \
        automake \
        autoconf \
        libtool \
        wget \
        libpmi2-0-dev \
        ca-certificates \
	graphviz \
        libunwind-dev \
        libopenblas-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install setuptools==69.5.1 datasets accelerate evaluate tokenizers sentencepiece transformers nltk deepspeed flash-attn nvtx \
    tqdm protobuf tensorboard tiktoken wandb drawsvg gurobipy pulp scipy pyarrow regex

RUN wget https://ftp.gnu.org/gnu/autoconf/autoconf-2.71.tar.xz && tar -xf autoconf-2.71.tar.xz && cd autoconf-2.71 && ./configure && make && make install && cd .. && rm autoconf-2.71.tar.xz
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
        libfftw3-dev \
        protobuf-compiler \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# DLRM & Chakra related
RUN git clone https://github.com/mlperf/logging.git mlperf-logging && pip install -e mlperf-logging && \
    pip install https://github.com/mlcommons/chakra/archive/refs/heads/main.zip && git clone https://github.com/facebookresearch/param.git && \
    cd param/et_replay && git checkout 7b19f586dd8b267333114992833a0d7e0d601630 && pip install . && \
    cd ../../ && git clone https://github.com/facebookresearch/HolisticTraceAnalysis.git && \
    cd HolisticTraceAnalysis && git checkout cc4ce6098bc73800b43a131a6d1e986b82d54230 && git submodule update --init && \
    pip install -r requirements.txt && \
    # Insert the following line to the file hta/common/trace.py at line 185
    # df["stream"] = df["stream"].astype(int)
    sed -i '185i\    df["stream"] = df["stream"].astype(int)' hta/common/trace.py && \
    pip install -e .

RUN pip install setuptools==69.5.1 datasets accelerate evaluate tokenizers sentencepiece transformers nltk deepspeed flash-attn nvtx \
    tqdm protobuf tensorboard tiktoken wandb drawsvg gurobipy pulp scipy pyarrow regex

RUN wget https://ftp.gnu.org/gnu/autoconf/autoconf-2.71.tar.xz && tar -xf autoconf-2.71.tar.xz && cd autoconf-2.71 && ./configure && make && make install && cd .. && rm autoconf-2.71.tar.xz

# Copy the entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
# Make the entrypoint script executable
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
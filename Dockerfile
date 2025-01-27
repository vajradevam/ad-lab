FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    python3-venv \
    pandoc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl https://pyenv.run | bash
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

RUN pyenv install 3.10.9 && pyenv global 3.10.9

RUN python3 -m venv /root/venv

RUN /root/venv/bin/pip install --upgrade pip && \
    /root/venv/bin/pip install \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    jupyterlab \
    notebook \
    jupyter \
    tensorflow \
    torch \
    matplotlib \
    seaborn

ENV VIRTUAL_ENV="/root/venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

EXPOSE 8888

CMD ["jupyter-lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]


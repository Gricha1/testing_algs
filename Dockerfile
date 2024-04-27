FROM continuumio/miniconda3

ADD environment.yaml /tmp/
RUN conda env create -f /tmp/environment.yaml

WORKDIR /usr/home/workspace

SHELL ["conda", "run", "-n", "HRAC", "/bin/bash", "-c"]



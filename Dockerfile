FROM continuumio/miniconda3:4.7.12

ADD ./environment.yml ./environment.yml

RUN conda env update -n ensambler -f ./environment.yml
ENV PATH /opt/conda/envs/ensambler/bin:$PATH
RUN /bin/bash -c "source activate ensambler"

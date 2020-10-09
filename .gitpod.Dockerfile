FROM gitpod/workspace-full-vnc:branch-jx-python-tk


USER root

RUN apt-get update && \
    apt-get -y install graphviz  && \
    apt-get clean && \
    apt-get -y autoremove



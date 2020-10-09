FROM gitpod/workspace-full-vnc:branch-jx-python-tk


USER root

RUN apt-get update \
 && apt-get install python3-pydot graphviz \
 && apt-get clean  && rm -rf /var/cache/apt/* && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*    



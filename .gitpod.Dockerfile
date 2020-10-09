FROM gitpod/workspace-full-vnc:branch-jx-python-tk


USER root

RUN apt-get update \
 && apt-get -y install graphviz 
    
    
    
# Cleaning
RUN apt-get clean  && rm -rf /var/cache/apt/* && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*    



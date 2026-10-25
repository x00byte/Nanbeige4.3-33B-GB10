FROM eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595
COPY patches /opt/nanbeige-patches
RUN python3 /opt/nanbeige-patches/apply.py
LABEL io.x00byte.nanbeige.recipe="flash-prefill-v3"

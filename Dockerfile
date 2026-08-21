FROM pytorch/pytorch:2.13.0-cuda13.2-cudnn9-devel

WORKDIR /workspace

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Lightricks/LTX-2.git /workspace/LTX-2
WORKDIR /workspace/LTX-2
RUN pip install -e packages/ltx-pipelines --break-system-packages || pip install -e packages/ltx-pipelines
RUN pip install runpod boto3 requests --break-system-packages || pip install runpod boto3 requests

COPY handler.py /workspace/handler.py
WORKDIR /workspace

CMD ["python3", "-u", "/workspace/handler.py"]

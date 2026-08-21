FROM runpod/pytorch:2.13.0-py3.12-cu132-devel

WORKDIR /workspace

RUN git clone https://github.com/Lightricks/LTX-2.git /workspace/LTX-2
WORKDIR /workspace/LTX-2
RUN pip install -e packages/ltx-pipelines --break-system-packages
RUN pip install runpod boto3 requests --break-system-packages

COPY handler.py /workspace/handler.py
WORKDIR /workspace

CMD ["python3", "-u", "/workspace/handler.py"]

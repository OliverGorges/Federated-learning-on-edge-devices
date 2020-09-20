FROM tensorflow/tensorflow:2.2.0
EXPOSE 5000
EXPOSE 6006

COPY . /app
ENV PYTHONPATH="/app:/app/tfmodels:/app/tfmodels/research:/app/tfmodels/official:/app/tfmodels/research/slim:${PYTHONPATH}"

RUN apt-get install -y protobuf-compiler libsm6 libxext6 libxrender-dev
COPY ./aws/ /root/.aws/

WORKDIR /app/tfmodels/research
RUN echo "$PWD"
RUN protoc object_detection/protos/*.proto --python_out=.
RUN pip install .
WORKDIR /app
RUN mkdir tasks checkpoints
RUN pip install -r requirements.txt

RUN tensorboard --logdir /app/models/ --host 0.0.0.0 &
CMD python /app/tffserver.py
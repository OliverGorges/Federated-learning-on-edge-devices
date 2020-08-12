FROM tensorflow/tensorflow:2.2.0
COPY . /app
EXPOSE 6006

ENV PYTHONPATH="/app:/app/tfmodels:/app/tfmodels/research:/app/tfmodels/official:/app/tfmodels/research/slim:${PYTHONPATH}"

RUN apt-get install -y protobuf-compiler libsm6 libxext6 libxrender-dev

WORKDIR /app/tfmodels/research
RUN protoc object_detection/protos/*.proto --python_out=.
RUN pip install .

WORKDIR /app/Traindata
RUN mkdir data
RUN mkdir output
WORKDIR /app/Traindata/data
RUN mkdir images
RUN mkdir annotations

WORKDIR /app

COPY ./aws/ /root/.aws/
RUN ls -l /root/.aws

RUN mkdir tasks
RUN mkdir checkpoints

RUN pip install -r requirements.txt
CMD python /app/trainer.py

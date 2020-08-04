FROM tensorflow/tensorflow:2.2.0
COPY . /app
ENV PYTHONPATH="/app:/app/tfmodels/models:/app/tfmodels/models/research:/app/tfmodels/models/official:/app/tfmodels/models/research/slim:${PYTHONPATH}"
WORKDIR /app/tfmodels/models/research
RUN apt-get install -y protobuf-compiler libsm6 libxext6 libxrender-dev mongodb
#RUN mv /app/aws/* ~/.aws/
RUN echo "$PWD"
RUN protoc object_detection/protos/*.proto --python_out=.
RUN pip install .
WORKDIR /app
RUN pip install -r requirements.txt
CMD python /app/tffserver.py
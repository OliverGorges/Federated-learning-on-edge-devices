FROM tensorflow/tensorflow:1.14.0-gpu-py3
COPY . /app
ENV PYTHONPATH="/app:/app/tfmodels/research:/app/tfmodels/research/slim:${PYTHONPATH}"
WORKDIR /app
CMD pip install -r requirements.txt
CMD protoc tfmodels/research/object_detection/protos/*.proto --python_out=.
CMD python /app/trainModel.py
FROM tensorflow/tensorflow:1.13.2-gpu-py3
COPY . /app
ENV PYTHONPATH="/app:/app/tfmodels/research:/app/tfmodels/research/slim:${PYTHONPATH}"
WORKDIR /app
RUN pip install -r requirements.txt
RUN mv /app/aws/* ~/.aws/
CMD protoc /app/tfmodels/research/object_detection/protos/*.proto --python_out=.
CMD python /app/trainModel.py
FROM tensorflow/tensorflow:2.2.0
COPY . /app
ENV PYTHONPATH="/app:/app/tfmodels/research:/app/tfmodels/research/slim:${PYTHONPATH}"
WORKDIR /app
RUN pip install -r requirements.txt
RUN mv /app/aws/* ~/.aws/
CMD protoc /app/tfmodels/research/object_detection/protos/*.proto --python_out=.
CMD python /app/rffserver.py
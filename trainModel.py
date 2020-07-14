from object_detection.dataset_tools.create_coco_tf_record import _create_tf_record_from_coco_annotations
from utils.Dataset.cocoAnnotationConverter import  XmlConverter, JsonConverter
from utils.Dataset import pipeline_config
import os
from shutil import copy
import pathlib
import tensorflow as tf
import boto3
from botocore.config import Config
from zipfile import ZipFile
import time
import logging 
from object_detection import model_hparams
from object_detection import model_lib

logging.basicConfig(level=logging.INFO)

#TF GPUConfig
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

CASE = 'MaskDetection'
DATA = 'Images'
Format = 'XML'

#Prepare Data
imageDir = os.path.join("Dataset", CASE , DATA)
outputDir = os.path.join("Traindata", "data")
model = "ssd_mobilenet_v2_coco_2018_03_29"
model_dir = os.path.join("Traindata", "model", model)
checkpoint_dir = os.path.join(model_dir)
for img in os.listdir(imageDir):
    copy(os.path.join(imageDir,img), os.path.join(outputDir, "images"))

# Create Pipeline Config
config = {}
config["label_map"] = str(pathlib.Path(os.path.join("Traindata", "data", "labelmap.pbtxt")).absolute())
config["checkpoint"] = str(pathlib.Path(os.path.join(checkpoint_dir,  "model.ckpt")).absolute())

# Convert single annotation Files to AnnotationFile for Train and Eval
if Format == "XML":
    annotationFiles, size =  XmlConverter().convert(os.path.join(outputDir, "images"), os.path.join("Dataset", CASE, "Annotations"),  outputDir, os.path.join("Dataset", CASE, "label_map.xml"), 0.7)
else: 
    annotationFiles, size =  JsonConverter().convert(os.path.join(outputDir, "images"), os.path.join("Dataset", "Annotations"),  outputDir, None, 0.7)
print( annotationFiles)

# Creates TFrecord for Train and Eval Annotations
for annotationFile in annotationFiles:
    name = str(os.path.basename(annotationFile)).split(".")[0]
    if "Train" in name:
        record = os.path.join(outputDir, "tfrecords", 'Train.record')
        _create_tf_record_from_coco_annotations(annotationFile, os.path.join(outputDir, "images"), record, False, 1)
        config["train_input"] = str(pathlib.Path(record).absolute())+"-00000-of-00001"
    elif "Eval" in name:
        record = os.path.join(outputDir, "tfrecords", 'Eval.record')
        _create_tf_record_from_coco_annotations(annotationFile, os.path.join(outputDir, "images"), record, False, 1)
        config["eval_input"] = str(pathlib.Path(record).absolute())+"-00000-of-00001"
        config["num_eval"] = size[1]

# Modify Pipeline file with the configuration data
pipeline_config.setConfig(os.path.join(model_dir, "pipeline.config"), config, os.path.join(model_dir, "custom_pipeline.config"))
outDir = "Traindata/model/FaceDetect/"

# Prepare Training
train_config = tf.estimator.RunConfig(outDir)
train_and_eval_dict = model_lib.create_estimator_and_inputs(
    run_config=train_config,
    hparams=model_hparams.create_hparams(None),
    pipeline_config_path=os.path.join(model_dir, "custom_pipeline.config"),
    train_steps=10000,
    sample_1_of_n_eval_examples=1,
    sample_1_of_n_eval_on_train_examples=(5))
estimator = train_and_eval_dict['estimator']
train_input_fn = train_and_eval_dict['train_input_fn']
eval_input_fns = train_and_eval_dict['eval_input_fns']
eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
predict_input_fn = train_and_eval_dict['predict_input_fn']
train_steps = train_and_eval_dict['train_steps']

if False:
    if False:
        name = 'training_data'
        input_fn = eval_on_train_input_fn
    else:
        name = 'validation_data'
        # The first eval input will be evaluated.
        input_fn = eval_input_fns[0]
    if False:
        estimator.evaluate(input_fn,
                            steps=10,
                            checkpoint_path=tf.train.latest_checkpoint(checkpoint_dir))
    else:
        model_lib.continuous_eval(estimator, model_dir, input_fn,train_steps, name)
else:
    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=False)

# Currently only a single Eval Spec is allowed.
tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])



# Upload Results to S3
checkpoints = sorted(os.listdir(outdir))[-3:]
zip_file = os.path.join(outDir, 'ckpt.zip')
with ZipFile(zip_file, 'w') as zip:
    for c in checkpoints:
        zip.write(os.path.join(outDir, c))
    zip.close()
try:
    s3 = boto3.resources('s3')
    s3.Bucket('federatedlearning').upload_file(zip_file, f'checkpoints/ckpt{time.time()}.zip')
except:
    logging.info("Cant upload results")
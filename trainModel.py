import object_detection as od
import utils.Tensorflow.trainer
import utils.Dataset.cocoAnnotationConverter as cocoAnnotation
import os

#Prepare Data
imageDir = os.path.join("Dataset", "ThermalImages"
outputDir = os.path.join("Traindata", "data")
annotationFile = cocoAnnotation.convert(imageDir, os.path.join("Dataset", "Annotations"), None, ))
od.dataset_tools.create_coco_tf_record._create_tf_record_from_coco_annotations(annotationFile, imageDir, outputDir)
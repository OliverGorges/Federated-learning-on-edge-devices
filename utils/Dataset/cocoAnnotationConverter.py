import json
import os
import logging
from datetime import datetime
import pathlib

# Coco Dataformat
#https://cocodataset.org/#format-data


def convert(imageDir, annotationDir, labelmap, outputPath ):
  annoationFiles = os.listdir(os.path.join(annotationDir))

  coco = {}
  images = []
  annoations = []
  categories = []

  nrOfAnnotations = 0

  categories.append({
      "supercategory": "none",
      "name": "face",
      "id": 0
    })

  for i, fname in enumerate(annoationFiles):
    with open(os.path.join(annoationFiles, fname)) as json_file:
      anno = json.load(json_file)
      imageHeight = 60*8
      imageWidth = 80*8
      images.append({
        "file_name": pathlib.PATH(os.path.join(imageDir, anno["thermalImage"])).absolute(),
        "height": imageHeight,
        "width": imageWidth,
        "id": i
      })
      for obj in anno["objects"]:
        obj = {}
        obj["id"] = nrOfAnnotations
        obj["bbox"] = [anno["bbox"]["xmin"] * imageWidth, anno["bbox]"["ymin"] * imageHeight, (anno["bbox"]["xmax"] - anno["bbox"]["xmin"]) * imageWidth, (anno["bbox"]["ymax"] - anno["bbox"]["ymin"]) * imageHeight]
        obj["image_id"] = i
        obj["iscrowd"] = 1
        obj["category_id"] = 0


  coco["info"] = {
    "description": "Thermal FaceDetection",
    "version": "1.0",
    "year": 2020,
    "contributor": "Oliver Gorges",
    "date_created": now.strftime("%d/%m/%Y")
  }

  coco["type"] = "instances"
  coco["images"] = images
  coco["annoations"] = annoations
  coco["categories"] = categories
  with open(os.path.join(outputPath, f'coco{now.strftime("%d/%m/%Y")}.json'), 'w') as outfile:
      json.dump(annotation, outfile)

if __name__ == "__main__":
  convert(os.path.join("..", "..", "Dataset", "Images"), os.path.join("..", "..", "Dataset", "Annotations"), None, os.path.join("..", "..", "Dataset")  )    
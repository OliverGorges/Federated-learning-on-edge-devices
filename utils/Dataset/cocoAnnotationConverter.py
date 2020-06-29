import json
import os
import logging
from datetime import datetime
import pathlib

# Coco Dataformat
#https://cocodataset.org/#format-data


def convert(imageDir, annotationDir, labelmap, outputPath, split=0.7):
  annotationFiles = os.listdir(os.path.join(annotationDir))
  if not split == 1:
    sub = [annotationFiles[0:int(len(annotationFiles)*split)], annotationFiles[int(len(annotationFiles)*split):]]
    train = createAnnotation(imageDir, sub[0], annotationDir, labelmap, outputPath, "Train")
    eval = createAnnotation(imageDir, sub[1], annotationDir, labelmap, outputPath, "Eval")
    return [train, eval], [len(sub[0]), len(sub[1])]
  else:
    train = createAnnotation(imageDir, annotationFiles, annotationDir, labelmap, outputPath, "Train")
    eval = ""
    return [train, eval], [len(annotationFiles), 0]

def createAnnotation(imageDir, annotationFiles, annotationDir, labelmap, outputPath, tag="Train"):
  coco = {}
  images = []
  annotations = []
  categories = []

  nrOfAnnotations = 0

  categories.append({
      "supercategory": "none",
      "name": "face",
      "id": 0
    })

  for i, fname in enumerate(annotationFiles):
    with open(os.path.join(annotationDir, fname)) as json_file:
      anno = json.load(json_file)
      imageHeight = 60*8
      imageWidth = 80*8
      images.append({
        "file_name": str(pathlib.Path(os.path.join(imageDir, anno["thermalImage"])).absolute()),
        "height": imageHeight,
        "width": imageWidth,
        "id": i
      })
      for box in anno["objects"]:
        obj = {}
        obj["id"] = nrOfAnnotations
        box = [int(box["bbox"]["xmin"] * imageWidth), int(box["bbox"]["ymin"] * imageHeight), int((box["bbox"]["xmax"] - box["bbox"]["xmin"]) * imageWidth), int((box["bbox"]["ymax"] - box["bbox"]["ymin"]) * imageHeight)]
        obj["bbox"] = box
        obj["area"] = box[2] * box[3]
        obj["image_id"] = i
        obj["iscrowd"] = 1 if len(anno["objects"]) > 1 else 0
        obj["category_id"] = 0
        annotations.append(obj)
        nrOfAnnotations += 1


  coco["info"] = {
    "description": "Thermal FaceDetection",
    "version": "1.0",
    "year": 2020,
    "contributor": "Oliver Gorges",
    "date_created": datetime.now().strftime("%d/%m/%Y")
  }

  coco["type"] = "instances"
  coco["images"] = images
  coco["annotations"] = annotations
  coco["categories"] = categories
  output = os.path.join(outputPath, f'coco{tag}{datetime.now().strftime("%d%m%Y")}.json')
  with open(output, 'w') as outfile:
      json.dump(coco, outfile)

  return output
  

if __name__ == "__main__":
  convert(os.path.join("..", "..", "Dataset", "Images"), os.path.join("..", "..", "Dataset", "Annotations"), None, os.path.join("..", "..", "Traindata", "data", "tfrecords")  )
  print(f'Saved in {pathlib.Path(os.path.join("..", "..", "Dataset")).absolute()}')
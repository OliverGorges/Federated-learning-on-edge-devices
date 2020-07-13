import json
import xml.etree.ElementTree as ET
import os
import logging
from datetime import datetime
import pathlib

# Coco Dataformat
#https://cocodataset.org/#format-data

class JsonConverter():

  def convert(self, imageDir, annotationDir, outputPath, labelmap=None, split=0.7):
    annotationFiles = os.listdir(os.path.join(annotationDir))
    if not split == 1:
      sub = [annotationFiles[0:int(len(annotationFiles)*split)], annotationFiles[int(len(annotationFiles)*split):]]
      train = self.createAnnotation(imageDir, sub[0], annotationDir, labelmap, outputPath, "Train")
      eval = self.createAnnotation(imageDir, sub[1], annotationDir, labelmap, outputPath, "Eval")
      return [train, eval], [len(sub[0]), len(sub[1])]
    else:
      train = self.createAnnotation(imageDir, annotationFiles, annotationDir, labelmap, outputPath, "Train")
      eval = ""
      return [train, eval], [len(annotationFiles), 0]

  def createAnnotation(self, imageDir, annotationFiles, annotationDir, labelmap, outputPath, tag="Train"):
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


class XmlConverter():


  def convert(self, imageDir, annotationDir, outputPath, labelmap=None, split=0.7):
    annotationFiles = os.listdir(os.path.join(annotationDir))
    if not split == 1:
      sub = [annotationFiles[0:int(len(annotationFiles)*split)], annotationFiles[int(len(annotationFiles)*split):]]
      train = self.createAnnotation(imageDir, sub[0], annotationDir, labelmap, outputPath, "Train")
      eval = self.createAnnotation(imageDir, sub[1], annotationDir, labelmap, outputPath, "Eval")
      return [train, eval], [len(sub[0]), len(sub[1])]
    else:
      train = self.createAnnotation(imageDir, annotationFiles, annotationDir, labelmap, outputPath, "Train")
      eval = ""
      return [train, eval], [len(annotationFiles), 0]


  def createAnnotation(self, imageDir, annotationFiles, annotationDir, labelmap, outputPath, tag="Train"):
    coco = {}
    images = []
    annotations = []
    categories = []
    
    
    nrOfAnnotations = 0
    labelDict = {}
    if labelmap is not None:
      labels = ET.parse(labelmap).getroot()
      for label in labels.findall('label'):
        labelDict[str(label.find('name').text)] = int(label.find('id').text)
        categories.append({
        "supercategory": label.find('supercategory').text,
        "name":  label.find('name').text,
        "id":  int(label.find('id').text)
      })
    else:
      labels["face"] = 0
      categories.append({
          "supercategory": "none",
          "name": "face",
          "id": 0
        })

    for i, fname in enumerate(annotationFiles):
      
      anno = ET.parse(os.path.join(annotationDir, fname)).getroot()
      images.append({
        "file_name": str(pathlib.Path(os.path.join(imageDir, anno.find("filename").text)).absolute()),
        "height": int(anno.find('size').find('height').text),
        "width": int(anno.find('size').find('width').text),
        "id": i
      })
      for detection in anno.findall('object'):
        box = detection.find("bndbox")
        obj = {}
        obj["id"] = nrOfAnnotations
        box = [int(box.find("xmin").text), int(box.find("ymin").text), int(box.find("xmax").text), int(box.find("ymax").text)]
        obj["bbox"] = box
        obj["area"] = box[2] * box[3]
        obj["image_id"] = i
        obj["iscrowd"] = 1 if len(anno.findall('object')) > 1 else 0
        obj["category_id"] = labelDict[detection.find('name').text]
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
  #JsonConverter().convert(os.path.join("..", "..", "Dataset", "Images"), os.path.join("..", "..", "Dataset", "Annotations"), None, os.path.join("..", "..", "Traindata", "data", "tfrecords")  )
  
  folder = os.path.join(os.sep,"Users","olgorges","Downloads","667889_1176415_bundle_archive")
  XmlConverter().convert( os.path.join(folder,"images"), os.path.join(folder,"annotations"), os.path.join(folder,"output"), os.path.join(folder,"label_map.xml") )
  print(f'Saved in {pathlib.Path(os.path.join("..", "..", "Dataset")).absolute()}')
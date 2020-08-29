"""
Classes to Convert different kind of Foarmats into the Coco Dataformat used in Modeltraining

Coco Dataformat:
https://cocodataset.org/#format-data
"""

import json
import xml.etree.ElementTree as ET
import os
import logging
from datetime import datetime
import pathlib



class JsonConverter():
  """
  Converter from Single Json annotation file to Coco annoations
  """

  def convert(self, imageDir, annotationDir, outputPath, labelmap=None, split=0.7):
    """
    Entry method
    imageDir: Folder with imagefiles
    annotationDir: Folder with annotations files in json format
    outputPath: Folder where the new annotationfiles will be stored
    labelmap: Lablemap in jsonformat TODO: Does not convert to pbtxt, nedds to be added
    split: Train Eval Split
    """
    annotationFiles = os.listdir(os.path.join(annotationDir))
    if not split == 1:
      sub = [annotationFiles[0:int(len(annotationFiles)*split)], annotationFiles[int(len(annotationFiles)*split):]]
      train, train_categories = self.createAnnotation(imageDir, sub[0], annotationDir, labelmap, outputPath, "Train")
      eval, train_categories = self.createAnnotation(imageDir, sub[1], annotationDir, labelmap, outputPath, "Eval")
      return [train, eval], [len(sub[0]), len(sub[1])], train_categories
    else:
      train, train_categories = self.createAnnotation(imageDir, annotationFiles, annotationDir, labelmap, outputPath, "Train")
      eval = ""
      return [train, eval], [len(annotationFiles), 0], train_categories

  def createAnnotation(self, imageDir, annotationFiles, annotationDir, labelmap, outputPath, tag="Train"):
    """
    Method to creat Annoation File
    imageDir: Folder with imagefiles
    annotationDir: Folder with annotations files in json format
    outputPath: Folder where the new annotationfiles will be stored
    labelmap: Lablemap in jsonformat TODO: Does not convert to pbtxt, nedds to be added
    tag: Train, Eval, Test
    """
    coco = {}
    images = []
    annotations = []
    categories = []

    nrOfAnnotations = 0
    # Write Labelmap in Annoation file and creat dict for labelmapping
    # Default = one label for Face
    labelDict = {}
    if labelmap is not None:
      with open(labelmap) as json_file:
        labels = json.load(json_file)
        for label in labels['item']:
          labelDict[str(label['name'])] = int(label['id'])
          categories.append({
          "supercategory": label['supercategory'],
          "name":  label['name'],
          "id":  int(label['id'])
      })
    else:
      labelDict["Face"] = 1
      categories.append({
          "supercategory": "none",
          "name": "Face",
          "id": 1
        })

    for i, fname in enumerate(annotationFiles):
      with open(os.path.join(annotationDir, fname)) as json_file:
        print(os.path.join(annotationDir, fname))
        anno = json.load(json_file)
        imageHeight = anno.get('height', 480) 
        imageWidth = anno.get('width', 640)
        # Add all images
        images.append({
          "file_name": str(pathlib.Path(os.path.join(imageDir, anno["thermalImage"])).absolute()),
          "height": imageHeight,
          "width": imageWidth,
          "id": i
        })
        # Add all objects
        for box in anno["objects"]:
          obj = {}
          obj["id"] = nrOfAnnotations
          obj["category_id"] = labelDict[box["type"]]
          x = int(box["bbox"]["xmin"] * imageHeight)
          y = int(box["bbox"]["ymin"] * imageWidth)
          width = int(box["bbox"]["xmax"]* imageHeight) - x
          height = int(box["bbox"]["ymax"] * imageWidth) - y
          box = [x, y, width, height]
          obj["bbox"] = box
          obj["area"] = box[2] * box[3]
          obj["image_id"] = i
          obj["iscrowd"] = 1 if len(anno["objects"]) > 1 else 0
          
          annotations.append(obj)
          nrOfAnnotations += 1

    # Add Dataset info
    coco["info"] = {
      "description": "Thermal FaceDetection",
      "version": "1.0",
      "year": 2020,
      "contributor": "Oliver Gorges",
      "date_created": datetime.now().strftime("%d/%m/%Y")
    }

    # Write AnnotaionFile
    coco["type"] = "instances"
    coco["images"] = images
    coco["annotations"] = annotations
    coco["categories"] = categories
    output = os.path.join(outputPath, f'coco{tag}{datetime.now().strftime("%d%m%Y")}.json')
    with open(output, 'w') as outfile:
        json.dump(coco, outfile)

    # Write Tensorflow Labelmap
    with open(os.path.join(outputPath, 'labelmap.pbtxt'), 'w') as the_file:
        for c in categories:
          the_file.write('item\n')
          the_file.write('{\n')
          the_file.write('id :{}'.format(int(c['id'])))
          the_file.write('\n')
          the_file.write("name :'{0}'".format(str(c['name'])))
          the_file.write('\n')
          the_file.write('}\n')

    return output, len(categories)


class XmlConverter():
  """
  Converter from Single XML annotation file to Coco annoations
  """

  def convert(self, imageDir, annotationDir, outputPath, labelmap=None, split=0.7):
    """
    Entry method
    imageDir: Folder with imagefiles
    annotationDir: Folder with annotations files in XML format
    outputPath: Folder where the new annotationfiles will be stored
    labelmap: Lablemap in XML format TODO: Does not convert to pbtxt, nedds to be added
    split: Train Eval Split
    """
    annotationFiles = os.listdir(os.path.join(annotationDir))
    if not split == 1:
      sub = [annotationFiles[0:int(len(annotationFiles)*split)], annotationFiles[int(len(annotationFiles)*split):]]
      train, train_categories = self.createAnnotation(imageDir, sub[0], annotationDir, labelmap, outputPath, "Train")
      eval, eval_categories = self.createAnnotation(imageDir, sub[1], annotationDir, labelmap, outputPath, "Eval")
      return [train, eval], [len(sub[0]), len(sub[1])], train_categories
    else:
      train, train_categories = self.createAnnotation(imageDir, annotationFiles, annotationDir, labelmap, outputPath, "Train")
      eval = ""
      return [train, eval], [len(annotationFiles), 0], train_categories


  def createAnnotation(self, imageDir, annotationFiles, annotationDir, labelmap, outputPath, tag="Train"):
    """
    Method to creat Annoation File
    imageDir: Folder with imagefiles
    annotationDir: Folder with annotations files in json format
    outputPath: Folder where the new annotationfiles will be stored
    labelmap: Lablemap in jsonformat TODO: Does not convert to pbtxt, nedds to be added
    tag: Train, Eval, Test
    """
    coco = {}
    images = []
    annotations = []
    categories = []
    
    
    nrOfAnnotations = 500
    labelDict = {}

    # Write Labelmap in Annoation file and creat dict for labelmapping
    # Default = one label for Face
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
      labelDict["Face"] = 1
      categories.append({
          "supercategory": "none",
          "name": "Face",
          "id": 1
        })
    print(categories)
    # Read all files from the subset
    for i, fname in enumerate(annotationFiles):
      
      anno = ET.parse(os.path.join(annotationDir, fname)).getroot()
      # Adds all images
      images.append({
        "file_name": str(pathlib.Path(os.path.join(imageDir, anno.find("filename").text)).absolute()),
        "height": int(anno.find('size').find('height').text),
        "width": int(anno.find('size').find('width').text),
        "id": i + 500
      })
      # Adds Objects
      for detection in anno.findall('object'):
        box = detection.find("bndbox")
        obj = {}
        obj["id"] = nrOfAnnotations
        box = [int(box.find("xmin").text), int(box.find("ymin").text), int(box.find("xmax").text)-int(box.find("xmin").text) , int(box.find("ymax").text)-int(box.find("ymin").text)]
        obj["bbox"] = box
        obj["area"] = box[2] * box[3]
        obj["image_id"] = i  + 500
        obj["iscrowd"] = 1 if len(anno.findall('object')) > 1 else 0
        obj["category_id"] = labelDict[detection.find('name').text]
        annotations.append(obj)
        nrOfAnnotations += 1

    # Add Dataset Info
    coco["info"] = {
      "description": "Thermal FaceDetection",
      "version": "1.0",
      "year": 2020,
      "contributor": "Oliver Gorges",
      "date_created": datetime.now().strftime("%d/%m/%Y")
    }
    # Write AnnoationFile
    coco["type"] = "instances"
    coco["images"] = images
    coco["annotations"] = annotations
    coco["categories"] = categories
    output = os.path.join(outputPath, f'coco{tag}{datetime.now().strftime("%d%m%Y")}.json')
    with open(output, 'w') as outfile:
        json.dump(coco, outfile)
      
    # Write Tensorflow Labelmap
    with open(os.path.join(outputPath, 'labelmap.pbtxt'), 'a') as the_file:
        for c in categories:
          the_file.write('item\n')
          the_file.write('{\n')
          the_file.write('id :{}'.format(int(c['id'])))
          the_file.write('\n')
          the_file.write("name :'{0}'".format(str(c['name'])))
          the_file.write('\n')
          the_file.write('}\n')

    return output, len(categories)



if __name__ == "__main__":
  #JsonConverter().convert(os.path.join("..", "..", "Dataset", "Images"), os.path.join("..", "..", "Dataset", "Annotations"), None, os.path.join("..", "..", "Traindata", "data", "tfrecords")  )

  folder = os.path.join("Dataset", "ThermalFaceDetection")
  JsonConverter().convert( os.path.join(folder,"Images"), os.path.join(folder,"Annotations"), os.path.join(folder,"Output"),None )
  print(f'Saved in {pathlib.Path(os.path.join("..", "..", "Dataset")).absolute()}')
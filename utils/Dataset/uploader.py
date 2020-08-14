import os
import logging
import zipfile
import shutil
from utils.Tensorflow.trainer import augmentData

case = "ThermalDetection2"
data = "ThermalImages"
minImages = 20
Dataset = os.path.join("Dataset", case)
tempDir = os.path.join("Dataset", "temp")
imgDir = os.path.join(Dataset, data)
annoDir = os.path.join(Dataset, "Annotations")

if __name__ == "__main__":
    
    if not os.path.exists(Dataset):
        logging.error(f"Cant find dataset: {Dataset}")
        exit()

    if not os.path.exists(tempDir):
        os.makedirs(tempDir)

    split = int(len(os.listdir(annoDir))/minImages)
    print(split)
    augImages, augAnnotations = augmentData(imgDir, annoDir, tempDir, split)

    # Upload Data to S3
    datasets = os.listdir(tempDir)
    for ds in datasets:
        zip_file = os.path.join(tempDir, f'{ds}.zip')
        folders = os.listdir(os.path.join(tempDir, ds))
        with zipfile.ZipFile(zip_file, 'w') as zip:
            for folder in folders:
                if os.path.isdir(os.path.join(tempDir, ds, folder)):
                    for data in os.listdir(os.path.join(tempDir, ds, folder)):
                        zip.write(os.path.join(tempDir, ds, folder, data), f'{folder}/{data}')
                else:
                    zip.write(os.path.join(tempDir, ds, folder), f'{folder}')
            zip.close()
        try:
            exit()
            s3 = boto3.resources('s3')
            s3.Bucket('federatedlearning').upload_file(zip_file, f'data/{ds}.zip')
        except:
            logging.info("Cant upload results")

    shutil.rmtree(os.path.join(tempDir))
        

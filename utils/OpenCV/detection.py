import cv2
import os 

class FaceDetection():

    def __init__(self):
        # Setup Opencv
        cascPath = os.path.join(os.path.dirname(__file__),"class.xml")
        self.faceCascade = cv2.CascadeClassifier(cascPath)

    def prepareImage(self, image, scale):
        print("Prepare Image")
        width, height = image.shape[:2]
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.cv2.resize(image, (int(height / scale), int(width / scale)))

        return self.image

    def detectFace(self, image=None, normalized=True):
        if not image:
            image = self.image
        print(image)
        self.faces = self.faceCascade.detectMultiScale(
                image,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
        )
        if normalized:
            return self.normalizeBoxes()
        else:
            return self.faces
    

    def normalizeBoxes(self, image=None, faces=None):
        if not image:
            width, height = self.image.shape[:2]
        else:
            width, height = image.shape[:2]
        if not faces:
            faces = self.faces

        normFaces = []
        for (x, y, w, h) in faces:
            normFaces.append(
                (
                    x / width,
                    y / height,
                    (x+w) / width,
                    (y+h) / height
                )
            )
        return normFaces

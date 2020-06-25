
from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_thermal_imaging import BrickletThermalImaging
import time
import logging

class Camera():

    setup = False
    hcImg = False
    tempImg = False

    def __init__(self, host="localhost", port=4223, uid="LcN"):
        logging.info("Start Connection")
        self.ipcon = IPConnection() # Create IP connection
        self.ti = BrickletThermalImaging(uid, self.ipcon) # Create device object
        self.ipcon.connect(host, port) # Connect to brickd
        while self.ipcon.get_connection_state() == 2 :
            logging.info(".")
        logging.debug(self.ipcon.get_connection_state())

    def getConnectionState(self):
        return self.ipcon.get_connection_state()


    def isTempImage(self):
        logging.info("Setup Temperatur Image")
        self.ti.set_image_transfer_config(self.ti.IMAGE_TRANSFER_MANUAL_TEMPERATURE_IMAGE)
        self.ti.set_resolution(self.ti.RESOLUTION_0_TO_655_KELVIN)
        time.sleep(0.5)
        self.setup = True

    def getTemperatureImage(self):
        if not self.ipcon.get_connection_state() == 1:
            __init__()

        if not self.setup or self.hcImg:
            self.isTempImage()
            self.tempImg = True
            self.hcImg = False
        return self.ti.get_temperature_image()

    def isHighContrastImage(self):
        logging.info("Setup High Contrast Image")
        self.ti.set_image_transfer_config(self.ti.IMAGE_TRANSFER_MANUAL_HIGH_CONTRAST_IMAGE)
        self.ti.set_resolution(self.ti.RESOLUTION_0_TO_655_KELVIN)
        time.sleep(0.5)
        self.setup = True

    def getHighContrastImage(self):
        if not self.ipcon.get_connection_state() == 1:
            __init__()

        if not self.setup or self.tempImg:
            self.isHighContrastImage()
            self.hcImg = True
            self.tempImg = False
        return self.ti.get_high_contrast_image()


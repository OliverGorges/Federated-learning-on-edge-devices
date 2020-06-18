
from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_thermal_imaging import BrickletThermalImaging
import time


class Camera():

    setup = False

    def __init__(self, host="localhost", port=4223, uid="LcN"):
        print("start Connection")
        self.ipcon = IPConnection() # Create IP connection
        self.ti = BrickletThermalImaging(uid, self.ipcon) # Create device object
        self.ipcon.connect(host, port) # Connect to brickd
        while self.ipcon.get_connection_state() == 2 :
            print(".")
        print(self.ipcon.get_connection_state())

    def isTempImage(self):
        self.ti.set_image_transfer_config(self.ti.IMAGE_TRANSFER_MANUAL_TEMPERATURE_IMAGE)
        self.ti.set_resolution(self.ti.RESOLUTION_0_TO_655_KELVIN)
        time.sleep(0.5)

    def getTemperatureImage(self):
        if not self.setup or self.hcImg:
            self.isTempImage()
            self.tempImg = True
            self.hcImg = False
        return self.ti.get_temperature_image()

    def isHighContrastImage(self):
        self.ti.set_image_transfer_config(self.ti.IMAGE_TRANSFER_MANUAL_HIGH_CONTRAST_IMAGE)
        self.ti.set_resolution(self.ti.RESOLUTION_0_TO_655_KELVIN)
        time.sleep(0.5)

    def getHighContrastImage(self):
        if not self.setup or self.hcImg:
            self.isHighContrastImage()
            self.hcImg = True
            self.tempImg = False
        return self.ti.get_temperature_image()

    
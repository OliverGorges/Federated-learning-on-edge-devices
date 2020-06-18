import time 

class FPS():

    def __init__(self, refreshRate=2):
        self.fps = 0
        self.count = 0
        self.start = time.time()
        self.refresh = refreshRate

    def tick(self):
        # FPS Calculation
        self.end = time.time()
        if (self.end - self.start > self.refresh):
            self.fps = int(self.count / self.refresh)
            self.start = time.time()
            self.count = 0
        else:
            self.count += 1
        return self.fps
        
    def print(self):
        if self.count == 0 :
            print(self.fps)
    
    def __int__(self):
        return int(self.fps)
    
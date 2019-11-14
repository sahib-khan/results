#!/usr/bin/python

import vot
import sys
import time
import cv2
import numpy as np
import collections

class NCCTracker(object):

    def __init__(self, image, region):
	self.region = region
	self.bbox = (int(region.x), int(region.y), int(region.width), int(region.height))
	self.gray=image
	self.prevgray=image

    def track(self, image):
	
	p1 = (int(self.bbox[0]), int(self.bbox[1]))
    	p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))

	
	vis = image.copy()
        self.gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        h, w, _ = vis.shape
        flow = np.zeros((h, w, 1), np.float32)
	return self.region
        flow = cv2.calcOpticalFlowFarneback(self.prevgray, self.gray, flow, 0.5, 5, 15, 3, 5, 1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
	newflow = flow[int(p1[1]):int(p2[1]),int(p1[0]):int(p2[0]),:]
	fx= newflow[...,0]
        fy= newflow[...,1]
        xavg=np.average(fx)
        yavg=np.average(fy)
        xsum=0
        count=0
	return self.region
	for x in np.nditer(fx):
            if xavg<0:
                if x<=-2.0:
                    xsum+=x
                    count+=1
            else:
                if x>=2.0:
                  xsum+=x
                  count+=1
        if count>0:
          xsum/=count
        deltax =xsum


        ysum=0
        count=0
        for y in np.nditer(fy):
            if yavg<0:
                if y<=-2.0:
                    ysum+=y
                    count+=1
            else:
                if y>=2.0:
                  ysum+=y
                  count+=1
        if count>0:
          ysum/=count
        
	deltay= ysum
	self.bbox = (int(self.bbox[0]+deltax),int(self.bbox[1]+deltay), int(self.bbox[2]), int(self.bbox[3]))

	p1 = (int(self.bbox[0]), int(self.bbox[1]))
    	p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
	self.prevgray = self.gray
	
	return vot.Rectangle(int(self.bbox[0]), int(self.bbox[1]), int(self.bbox[2]),int(self.bbox[3]))
	

handle = vot.VOT("rectangle")
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
tracker = NCCTracker(image, selection)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    region= tracker.track(image)
    handle.report(region)


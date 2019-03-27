from edge import edge_boxes
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import myselectivesearch
import selectivesearch
import numpy as np
from skimage import feature as ft
import scipy

img = skimage.io.imread('C:\\Users\\lyming\\Desktop\\anomalydata\\anomalydata\\zoo\\KM1106+186M.jpg')
windows = edge_boxes.get_windows('C:\\Users\\lyming\\Desktop\\anomalydata\\anomalydata\\zoo\\KM1106+186M.jpg')
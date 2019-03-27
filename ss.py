from __future__ import (
    division,
    print_function,
)

import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import myselectivesearch
import selectivesearch
import numpy as np
from skimage import feature as ft
import scipy
def chisquare(x,y):
    subMatrix = x - y;
    subMatrix2 = subMatrix**2;
    addMatrix = x + y;

    addMatrix[addMatrix == 0]=1;
    DistMat = subMatrix2/ addMatrix;
    D = sum(DistMat, 2)/len(DistMat);
    return D;
def main():

    # loading astronaut image

    #img = skimage.io.imread('C:\\Users\\lyming\\Desktop\\anomalydata\\anomalydata\\zoo1\\animal1.jpg')
    img = skimage.io.imread('C:\\Users\\lyming\\Desktop\\anomalydata\\anomalydata\\zoo\\KM1106+186M.jpg')

    Mid = 397;
    img = np.dstack([img] * 3)
    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=100)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        #if r['size'] < 500 or r['size'] > 10000:
        #    continue

        # distorted rects
        x, y, w, h = r['rect']
        if w> 200 or h>200 or w <40 or h<40:
            continue
        if x>Mid+280 or x<Mid-280:
            continue
        if w*h > 200*200 or w*h <50*50:
            continue
        if w==0 or h==0:
            continue
        if w / h > 3 or h / w > 3:
            continue
        img1 = img[y:y+h,x:x+w,0]
        if x>=Mid:
            if Mid-(x-Mid)-w<0 or Mid-(x-Mid)>800:
                continue
            img2 = img[y:y+h,Mid-(x-Mid)-w:Mid-(x-Mid),0]
        else:
            if Mid + (Mid - x) - w < 0 or Mid + (Mid - x) > 800:
                continue
            img2 = img[y:y + h, Mid + (Mid - x) - w:Mid + (Mid - x),0]
        img2 = img2[:,::-1]
        feature1, hogmap1 = ft.hog(img1,  # input image
                                 orientations=9,  # number of bins
                                 pixels_per_cell=(8, 8),  # pixel per cell
                                 cells_per_block=(3, 3),  # cells per blcok
                                 block_norm='L1',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                                 transform_sqrt=True,  # power law compression (also known as gamma correction)
                                 feature_vector=True,  # flatten the final vectors
                                 visualise=True)  # return HOG map)
        feature2, hogmap2 = ft.hog(img2,  # input image
                                   orientations=9,  # number of bins
                                   pixels_per_cell=(8, 8),  # pixel per cell
                                   cells_per_block=(3, 3),  # cells per blcok
                                   block_norm='L1',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                                   transform_sqrt=True,  # power law compression (also known as gamma correction)
                                   feature_vector=True,  # flatten the final vectors
                                   visualise=True)  # return HOG map)
        hogchisquare = chisquare(feature1,feature2)#/w/h
        if(hogchisquare<0.01):
            continue
        #hogchisquare = 0
        #print(hogchisquare,r['size'],r['rect'])
        candidates.add((r['rect'],hogchisquare))

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    #ax.imshow(np.dstack([img] * 3))
    ax.imshow(img)
    for (x, y, w, h),hogscore in candidates:
        print(x, y, w, h,hogscore*1000)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.text(x, y, str(int(hogscore*1000)))
        ax.add_patch(rect)

    plt.show()

if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from PIL import Image
from numpy import asanyarray
import numpy as np
import os

def get_image_names(images_dir=".", filetype="png"):
    # get all file names from sample images directory
    fileNames = os.listdir(images_dir)
    
    # files <= only valid files + full path
    for i in range(len(fileNames) - 1, -1, -1):
        if not fileNames[i].endswith('.' + filetype) and not fileNames[i].split(".")[0].isdigit():
            del fileNames[i]
            
    fileNames.sort(key = lambda x: int(x.split(".")[0])) # sort based on numerical order

    fileNames = [ os.path.join(images_dir, x) for x in fileNames]

    return fileNames




if __name__ == "__main__":
    hue_threshold = 0.90
    value_threshold = 0.85

    imageNames = get_image_names()
    print(imageNames)
    
    imgs = []
    for name in imageNames:
        rgb_img = Image.open(name)
        rgb_img = asanyarray(rgb_img)
        rgb_img = rgb_img[:, :, 0:3]

        hsv_img = rgb2hsv(rgb_img)
        hue_img = hsv_img[:, :, 0]
        value_img = hsv_img[:, :, 2]

        mask = (hue_img >= hue_threshold) & (value_img >= value_threshold)
        masked_img = np.zeros_like(hue_img)
        masked_img[mask] = hue_img[mask]
        imgs.append(masked_img)
    
    img_arr = np.array(imgs)
    img_grid = img_arr.reshape(3, 3, *img_arr.shape[1:])

    fig, axis = plt.subplots(ncols=3, nrows=3, figsize=(8,2)) # we know there is 9 imgs

    for i in range(3):
        for j in range(3):
            axis[i, j].imshow(img_grid[i, j])
    

    plt.show()

import numpy as np
from skimage import io
import os

for root, dirs, files in os.walk("./masks", topdown=True):
    for image in files:
        if image.endswith(".png"):
            # print(image.split(sep=".p")[0])
            mask = io.imread(os.path.join(root, image), as_gray=True)
            dumy_var = image.split(sep="_")
            dumy_var = dumy_var[-1].split(".")[0]
            print(
                f"{dumy_var}, {np.sum(mask != 0.0) / (mask.shape[0] * mask.shape[1])}")

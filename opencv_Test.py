import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# import sys
# x = np.array([[2, 4, 6], [8, 10, 12], [9, 9, 9]])
# y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# print(x*y)
# print(x@y)
# sys.exit()

# From (0,0) -> (7,7)
block_range = [(i, j) for i in range(8) for j in range(8)]

# cosine table for the summation
cos_table = np.array([[np.cos((2*i+1)*j * np.pi/16) for j in range(8)] for i in range(8)], dtype=float)

quantization_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58,  60, 55],
    [14, 13, 16, 24, 40, 57,  69, 56],
    [14, 17, 22, 29, 51, 87,  80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104,  113, 92],
    [49, 64, 78, 87, 103, 121,  120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
])

root2_inverse = 1 / np.sqrt(2)



def dct(im_block, u, v):
    dct_coef = 0

    for i, j in block_range:
        dct_coef += im_block[i, j] * cos_table[i, u] * cos_table[j, v]

    if u == 0:
        dct_coef *= root2_inverse
    if v == 0:
        dct_coef *= root2_inverse
    dct_coef *= 0.25
    return dct_coef


def i_dct(im_block, i, j):
    inverse = 0
    for u, v in block_range:
        coef = im_block[u, v] * cos_table[i, u] * cos_table[j, v]
        if u == 0:
            coef *= root2_inverse
        if v == 0:
            coef *= root2_inverse
        inverse += coef
    inverse *= 0.25
    return int(np.round(inverse))


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:    # No noise
        return 100
    return 20 * np.log10(255 / np.sqrt(mse))

img = cv.imread(r"C:\Users\Saitako\PycharmProjects\pythonProject\img\cameraman.tif", cv.IMREAD_GRAYSCALE)

height, width = img.shape

# 8x8 block
block_height = int(height / 8)
block_width = int(width / 8)

new_im = np.zeros((width, height), dtype=int)

for i in range(block_height):
    for j in range(block_width):
        block = img[i*8:(i+1)*8, j*8:(j+1)*8]
        trans_im = np.array([dct(block, u, v) for u, v in block_range]).reshape((8, 8))
        quant_im = np.matrix.round(np.true_divide(trans_im, quantization_table))
        dequant_im = quant_im * quantization_table

        inv = np.array([i_dct(dequant_im, i, j) for i, j in block_range]).reshape((8, 8))
        new_im[i*8:(i+1)*8, j*8:(j+1)*8] = inv

cv.imwrite("./output/compressed.tif", new_im)
print(PSNR(img, new_im))


"""

for i in range(width):
        for j in range(height):
            if img[i, j] >= 225:
                img[i, j] = 255
def show_im(img):
    cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def filter_ignore_edge(img, constrain, f_offset, im_offset_01,
                       im_offset_02, filter, median_filter=False):
    height, width = img.shape
    new_img = np.copy(img)
    for i in range(height):
        if i in constrain:
            continue
        if i + f_offset == height:
            break
        for j in range(width):
            if j in constrain:
                continue
            if j + f_offset == width:
                break

            if not median_filter:
                temp = img[i - im_offset_01:i + im_offset_02, j - im_offset_01:j + im_offset_02] * filter
                new_img[i, j] = temp.sum()

            else:
                temp = img[i - im_offset_01:i + im_offset_02,
                       j - im_offset_01:j + im_offset_02].flatten()
                new_img[i, j] = np.median(temp)

    return new_img

img = cv.imread('./img/t_03.jpg', cv.IMREAD_GRAYSCALE)

# high-pass filters
Laplacian = np.array(([1, -2, 1], [-2, 4, -2], [1, -2, 1]), dtype=int)
h_filter = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]), dtype=int)

new_img = filter_ignore_edge(img=img, constrain=[0], f_offset=1, im_offset_01=1,
                             im_offset_02=2, filter=Laplacian)
cv.imwrite("./output/img/Laplacian.png", new_img)
            elif 225 > img[i, j] >= 200:
                img[i, j] = 212
            elif 200 > img[i, j] >= 175:
                img[i, j] = 187
            elif 175 > img[i, j] >= 150:
                img[i, j] = 162
            elif 150 > img[i, j] >= 125:
                img[i, j] = 137
            elif 125 > img[i, j] >= 100:
                img[i, j] = 112
            elif 100 > img[i, j] >= 50:
                img[i, j] = 75
            else:
                img[i, j] = 0

def filter_ignore_edge(img, constrain, f_offset, im_offset_01,
                       im_offset_02, filter, filter_type):
    height, width = img.shape
    new_img = np.copy(img)
    for i in range(height):
        if i in constrain:
            continue
        if i + f_offset == height:
            break
        for j in range(width):
            if j in constrain:
                continue
            if j + f_offset == width:
                break
            if filter_type == "mean":
                temp = img[i - im_offset_01:i + im_offset_02,
                       j - im_offset_01:j + im_offset_02] * filter
                new_img[i, j] = temp.sum()

            else:
                temp = img[i - im_offset_01:i + im_offset_02,
                       j - im_offset_01:j + im_offset_02].flatten()
                new_img[i, j] = np.median(temp)

    return new_img


def to_gray_scale(img):
    width, height, _ = img.shape
    temp_img = np.zeros((width, height), dtype=int)
    for i in range(width):
        for j in range(height):
            B = img.item(i, j, 0)
            G = img.item(i, j, 1)
            R = img.item(i, j, 2)
            temp_img[i, j] = 0.2989 * R + 0.587 * G + 0.1144 * B
    return temp_img


def show_im(img):
    cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
# Save the original image and histogram
plt.hist(img.ravel(), 256, [0, 256])
plt.savefig('./output/original_hist.png')
plt.cla()
cv.imwrite('./output/original_img.png', img)

hist, bins = np.histogram(img.flatten(), 256, [0, 256])
cdf = hist.cumsum()

for i in range(width):
    for j in range(height):
        img[i, j] = math.floor(cdf[img[i, j]] / img.size * 255)
        

# Save the processed image and histogram
hist_plot = plt.hist(img.ravel(), 256, [0, 256])
plt.savefig('./output/EQ_hist.png')
cv.imwrite('./output/EQ_im.png', img)


# 3x3 mean filters
mean_filter = np.ones((3, 3)) / 9
new_img = filter_ignore_edge(img=img, constrain=[0], f_offset=1, im_offset_01=1,
                             im_offset_02=2, filter=mean_filter, filter_type="mean")
cv.imwrite("./output/img/3x3_mean_filtered.png", new_img)

# 3x3 median filters
new_img = filter_ignore_edge(img=img, constrain=[0], f_offset=1, im_offset_01=1,
                             im_offset_02=2, filter=None, filter_type="median")
cv.imwrite("./output/img/3x3_median_filtered.png", new_img)  

# 5x5 mean filters
mean_filter = np.ones((5, 5)) / 25
new_img = filter_ignore_edge(img=img, constrain=[0, 1], f_offset=2, im_offset_01=2,
                             im_offset_02=3, filter=mean_filter, filter_type="mean")
cv.imwrite("./output/img/5x5_mean_filtered.png", new_img)

# 5x5 median filters
new_img = filter_ignore_edge(img=img, constrain=[0, 1], f_offset=2, im_offset_01=2,
                             im_offset_02=3, filter=None, filter_type="median")
cv.imwrite("./output/img/5x5_median_filtered.png", new_img)

# 9x9 mean filters
mean_filter = np.ones((9, 9)) / 81
new_img = filter_ignore_edge(img=img, constrain=[0, 1, 2, 3], f_offset=4, im_offset_01=4,
                             im_offset_02=5, filter=mean_filter, filter_type="mean")
cv.imwrite("./output/img/9x9_mean_filtered.png", new_img)

# 9x9 median filters
new_img = filter_ignore_edge(img=img, constrain=[0, 1, 2, 3], f_offset=4, im_offset_01=4,
                             im_offset_02=5, filter=None, filter_type="median")
# Bit plane
for k in range(8):
    n_img = np.copy(img)
    for i in range(width):
            for j in range(height):
                if int(f"{n_img[i, j]:0>8b}"[k]) == 1:
                    n_img[i, j] = 255

                else:
                    n_img[i, j] = 0

    cv.imwrite(f"./output/img/bit_plane_{k+1:2d}.png", n_img)

"""
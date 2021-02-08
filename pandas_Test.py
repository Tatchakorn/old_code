import numpy as np
import cv2

# read the image
image = cv2.imread('./img/lena.bmp')
cv2.imshow("output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# def meanFilter(im):
#     img = im
#     w = 2

#     for i in range(2, im.shape[0]-2):
#         for j in range(2, im.shape[1]-2):
#             block = im[i-w: i+w+1, j-w: j+w+1]
#             m = numpy.mean(block, dtype=numpy.float32)
#             img[i][j] = int(m)

#     return img


# def sp_noise(images):
#     r, c, _ = images[0].shape()
#     salt_vs_pepper = 0.2
#     amount = 0.003
#     num_salt = int(amount*images[0].size*salt_vs_pepper)
#     num_pepper = int(amount*images[0].size*(1.0 - salt_vs_pepper))

#     for image in images:
#         # add salt noise
#         cord = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape]
#         image[cord[0], cord[1], :] = 1

#         # add pepper noise
#         cord = [np.random.randint(0, i-1, int(num_pepper)) for i in image.shape]
#         image[cord[0], cord[1], :] = 0
#     return images

# img = meanFilter(image)
# cv2.imshow(img)
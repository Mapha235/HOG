import scipy.ndimage as nd
import scipy.signal
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize

import numpy as np
import cv2


def plot(img: np.ndarray, title: str):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.gray)
    plt.title(title)

# adds a black border around the image


def add_border(img: np.ndarray, border_size: int):
    height, width = img.shape
    border_img = np.zeros((height+2, width+2))

    for (x, y), value in np.ndenumerate(img):
        border_img[x+1][y+1] = value

    return border_img


def convolve_sobel(img: np.ndarray):
    sobel_kernel = np.array([[1,  0, -1],  # row 1
                             [-2, 0, -2],  # row 2
                             [1,  0, -1]])  # row 3

    kernel_x = np.array([[-1, 0, 1]])
    kernel_y = np.array([[-1],
                         [0],
                         [1]])

    res_x = scipy.signal.convolve2d(img, kernel_x, mode='same')
    res_y = scipy.signal.convolve2d(img, kernel_y, mode='same')

    return np.hypot(res_x,res_y)


def load_image(infilename):
    '''
    converts an image, that has been loaded with Pillow, into an ndarray
    '''
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def preprocess(img: np.ndarray, normalize=False):
    size = (64, 128)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def l2_norm():
    pass


def compute_gradients(img):
    '''
    Returns the horizontal and vertical image gradients
    '''
    kernel_x = np.array([[-1, 0, 1]])
    kernel_y = np.array([[-1],
                         [0],
                         [1]])

    grad_x = []
    grad_y = []

    height, width, channels = img.shape
    res_magn = np.zeros((height, width))
    res_ang = np.zeros((height, width))

    # compute horizontal and vertical gradients for each color channel
    for i in range(3):
        grad_x.append(scipy.signal.convolve2d(
            img[:, :, i], kernel_x, mode='same'))
        grad_y.append(scipy.signal.convolve2d(
            img[:, :, i], kernel_y, mode='same'))

    # check for the largest norm
    for i in range(height):
        for j in range(width):
            magnitudes = []
            for k in range(3):
                magnitudes.append(
                    np.sqrt(grad_x[k][i][j]**2 + grad_y[k][i][j]**2))

            res_magn[i][j] = max(magnitudes)
            # res_ang[i, j] =
    return np.hypot(grad_x[0], grad_y[0]), np.hypot(grad_x[1], grad_y[1]), np.hypot(grad_x[2], grad_y[2]), res_magn


def main():
    img = cv2.imread('./data/woman2.png')
    # plot(img, 'normal')

    # Create a black image
    img = np.zeros((640, 480, 3))
    # ... and make a white rectangle in it
    img[100: -100, 80: -80] = 1

    img = preprocess(img)
    plot(img, 'resized')

    grad_red, grad_green, grad_blue, grad_opt = compute_gradients(img)
    plot(grad_opt, 'grad opt')

    sx = nd.sobel(img, axis=0, mode='constant')
    sy = nd.sobel(img, axis=1, mode='constant')

    # plot(sx, 'Sobel X')
    # plot(sy, 'Sobel Y')

    # hypotenuse == root of squared sum == magnitude
    sobel = np.hypot(sx, sy)
    # print(sobel.dtype)
    # sobel = nd.sobel(img,axis=1, mode ='constant')
    # plot(sx, 'sobel x')
    # plot(sy, 'Sobel y')
    # plot((sobel).astype(np.uint8), 'SSD')
    plot(sobel, 'SSD')
    # plot(convolve_sobel(img), 'convolve')
    plt.show()


if __name__ == "__main__":
    main()

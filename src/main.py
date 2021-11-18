import scipy.ndimage as nd
import scipy.signal
import matplotlib.pyplot as plt
from PIL import Image

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

    res_x = scipy.signal.convolve2d(img[:, :, 0], kernel_x, mode='same')
    res_y = scipy.signal.convolve2d(img[:, :, 0], kernel_y, mode='same')

    return np.hypot(res_x, res_y)


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
    Returns the magnitudes and angles
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
            for k in range(channels):
                magnitudes.append(
                    np.sqrt(grad_x[k][i][j]**2 + grad_y[k][i][j]**2))

            index_max = max(range(len(magnitudes)), key=magnitudes.__getitem__)
            res_magn[i, j] = magnitudes[index_max]

            res_ang[i, j] = abs(np.arctan2(
                grad_y[index_max][i][j], grad_x[index_max][i][j]))
            res_ang[i, j] = (res_ang[i, j] * 360) / (2*np.pi)
            # res_ang[i, j] =
    # return np.hypot(grad_x[0], grad_y[0]), np.hypot(grad_x[1], grad_y[1]), np.hypot(grad_x[2], grad_y[2]), res_magn
    return res_magn, res_ang


def cells(img):
    x, y = (8, 8)


def orientation_binning(magnitudes, orientations, nr_of_bins=9, cell_size=8):
    y, x = magnitudes.shape

    if magnitudes.shape != orientations.shape or 2*x != y:
        return 1

    hist = np.zeros(nr_of_bins)
    steps_y = int(y / cell_size)
    steps_x = int(x / cell_size)
     
    for i in range(steps_x):
        for j in range(steps_y):
            # print(f"{i},{j}. ({i*cell_size},{(i+1)*cell_size})")
            temp = magnitudes[i:(i+1)*cell_size, j:(j+1)*cell_size]
            print(temp)
            


def main():
    img = cv2.imread('./data/man.png')
    # img = cv2.imread('./data/pedestrians.jpg')
    # plot(img, 'normal')

    # Create a black image
    # img = np.zeros((640, 480, 3))
    # # ... and make a white rectangle in it
    # img[100: -100, 80: -80] = 1

    img = preprocess(img)
    # plot(img, 'resized')

    # grad_red, grad_green, grad_blue, grad_mag = compute_gradients(img)
    grad_mag, grad_ang = compute_gradients(img)
    # plot(grad_mag, 'grad opt')
    # plot(grad_ang, 'grad ang')
    orientation_binning(grad_mag, grad_ang)

    # plot(convolve_sobel(img), 'convolve')
    plt.show()


if __name__ == "__main__":
    main()

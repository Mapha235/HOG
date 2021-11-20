from numpy.lib.function_base import interp
from skimage.io import imread
from skimage import draw
from skimage import exposure
from skimage.feature import hog
from skimage.transform import resize
from skimage.feature import _hoghistogram

from numpy.core.shape_base import block
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


# GRADIENT COMPUTATION------------------------------------------------------------------------------------------------


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
    return grad_x[0], grad_y[0], res_magn, res_ang

# ORIENTATION BINNING------------------------------------------------------------------------------------------------


def cells(magnitudes, orientations, nr_of_bins=9, cell_size=8):
    '''
    divides the image into cells of size cell_size x cell_size
    '''
    y, x = magnitudes.shape

    if magnitudes.shape != orientations.shape or 2*x != y:
        return 1

    hist = np.zeros(nr_of_bins)
    steps_y = int(y / cell_size)
    steps_x = int(x / cell_size)

    cells_mag = []
    cells_ang = []

    for i in range(steps_x):
        for j in range(steps_y):
            # print(f"{i},{j}. ({i*cell_size},{(i+1)*cell_size}), ({j*cell_size},{(j+1)*cell_size})")
            cells_mag.append(
                magnitudes[j*cell_size:((j+1)*cell_size), i*cell_size:((i+1)*cell_size)])
            cells_ang.append(
                orientations[j*cell_size:((j+1)*cell_size), i*cell_size:((i+1)*cell_size)])

    return cells_mag, cells_ang


def interpolate(magnitude: float, orientation: float, nr_of_bins=9, signed=True):
    '''
    Splits the magnitude value into left and right value according to their contribution to the adjacent bins.
    '''
    steps = 360 / nr_of_bins
    if signed:
        steps = steps / 2

    orientation = int(orientation)

    l_bin = int(orientation/steps) % 9
    r_bin = (int(orientation/steps) + 1) % 9

    l_vote = ((steps - orientation % steps) / steps) * magnitude
    r_vote = ((orientation % steps) / steps) * magnitude

    # print(f'{l_bin}, {l_vote}')
    # print(f'{r_bin}, {r_vote}')

    return (l_bin, l_vote), (r_bin, r_vote)


def binning(cells_magn, cells_ang):
    if len(cells_magn) != len(cells_ang):
        return 1

    histogram = np.zeros(shape=(len(cells_magn), 9))

    # histogram = np.zeros(9)

    # iterate over the cells
    for cell, ang, hist_index in zip(cells_magn, cells_ang, range(len(cells_magn))):
        if len(cell) != len(ang):
            return 1

        # loop through the cell values
        for i in range(len(cell)):
            for j in range(len(cell[i])):
                left, right = interpolate(
                    cell[i][j], ang[i][j], nr_of_bins=9, signed=True)

                histogram[hist_index][left[0]] += left[1]
                histogram[hist_index][right[0]] += right[1]

    return histogram.reshape(16, 8, 9)


def l2_norm(vector):
    '''
    Returns the L2-normalized vector.
    '''
    squared = [val ** 2 for val in vector]
    norm = np.sqrt(sum(squared))
    return [val / norm for val in vector]


def block_normalization(block_hists, nr_of_bins=9, stride=8):
    '''
    Returns the normalized
    '''
    rows, columns, bins = block_hists.shape
    # feature_descriptor = np.zeros(shape= ())
    feature_descriptor = []

    for row in range(rows-1):
        for column in range(columns-1):
            # block.append(np.concatenate([block_hists[row][column]  , block_hists[row][column+1],
            #                              block_hists[row+1][column], block_hists[row+1][column+1]]))
            block_norm = l2_norm(np.concatenate([block_hists[row][column], block_hists[row][column+1],
                                                 block_hists[row+1][column], block_hists[row+1][column+1]]))
            feature_descriptor = feature_descriptor + block_norm

    return feature_descriptor

# SKIIMAGE ----------------------------------------------------------------------------------------------------------


# VISUALIZE ---------------------------------------------------------------------------------------------------------
# copied from the scitkit-image library: https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/_hog.py

def visualize(g_row, g_col, orientation_histogram):

    hog_image = None
    # s_row, s_col = img.shape[:2]
    s_row, s_col = (128, 64)
    c_row, c_col = (8, 8)
    b_row, b_col = (2, 2)
    orientations = 9

    n_cells_row = int(s_row // c_row)  # number of cells along row-axis
    n_cells_col = int(s_col // c_col)  # number of cells along col-axis
    
    plt.figure()
    plt.bar(np.arange(0,9), orientation_histogram[0][0])

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cells_row, n_cells_col, orientations),
                                    dtype=float)
    g_row = g_row.astype(float, copy=False)
    g_col = g_col.astype(float, copy=False)

    _hoghistogram.hog_histograms(g_col, g_row, c_col, c_row, s_col, s_row,
                                n_cells_col, n_cells_row,
                                orientations, orientation_histogram)

    plt.figure()
    plt.bar(np.arange(0,9), orientation_histogram[0][0])
    plt.show()

    radius = min(c_row, c_col) // 2 - 1
    orientations_arr = np.arange(orientations)
    # set dr_arr, dc_arr to correspond to midpoints of orientation bins
    orientation_bin_midpoints = (
        np.pi * (orientations_arr + .5) / orientations)

    dr_arr = radius * np.sin(orientation_bin_midpoints)
    dc_arr = radius * np.cos(orientation_bin_midpoints)

    hog_image = np.zeros((s_row, s_col), dtype=float)

    for r in range(n_cells_row):
        for c in range(n_cells_col):
            for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                centre = tuple([r * c_row + c_row // 2,
                                c * c_col + c_col // 2])
                rr, cc = draw.line(int(centre[0] - dc),
                                   int(centre[1] + dr),
                                   int(centre[0] + dc),
                                   int(centre[1] - dr))
                hog_image[rr, cc] += orientation_histogram[r][c][o]

    return hog_image

# MAIN---------------------------------------------------------------------------------------------------------------


def main():
    img = cv2.imread('./data/b.png')
    # img = cv2.imread('./data/man.png')
    # img = cv2.imread('./data/pedestrians.jpg')
    # plot(img, 'normal')

    # Create a black image
    # img = np.zeros((640, 480, 3))
    # # ... and make a white rectangle in it
    # img[100: -100, 80: -80] = 1

    img = preprocess(img)
    # plot(img, 'resized')

    # grad_red, grad_green, grad_blue, grad_mag = compute_gradients(img)
    g_row, g_col, grad_mag, grad_ang = compute_gradients(img)

    plot(grad_mag, 'grad opt')
    # plot(grad_ang, 'grad ang')
    cells_mag, cells_ang = cells(grad_mag, grad_ang)
    hist = binning(cells_mag, cells_ang)

    plot(visualize(g_row, g_col, hist), 'HG')
    final_descr = block_normalization(hist)

    # free up memory
    grad_mag = []
    grad_ang = []
    cells_mag = []
    cells_ang = []
    hist = []

    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    plt.axis("off")
    plot(hog_image, 'HOG')
    
    # plt.figure()
    # plt.hist(final_descr)
    # plt.figure()
    # plt.hist(fd)

    plt.show()


if __name__ == "__main__":
    main()

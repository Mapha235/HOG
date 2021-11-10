import numpy as np
# from scipy import ndimage
import scipy.signal
import matplotlib.pyplot as plt


def convolve_sobel(img):
    sobel_kernel = [[1,  0, -1],
                    [-2, 0, -2],
                    [1,  0, -1]]
    
    res = scipy.signal.convolve2d(img, sobel_kernel, mode = 'same', boundary = 'fill', fillvalue = 0)
    return res

def plot(img, title):
    plt.figure()
    plt.imshow(img, cmap = plt.cm.gray)
    plt.title(title)

def main():
    # Create a black image
    img = np.zeros((640, 480))
    # ... and make a white rectangle in it
    img[100:-100, 80:-80] = 1

    # See how it looks
    plot(img, 'Original')
    plot(convolve_sobel(img), 'Sobel Convolution')
    plt.show()


if __name__ == "__main__":
    main()

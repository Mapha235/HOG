import numpy as np
from scipy import ndimage
import scipy.signal
import matplotlib.pyplot as plt


def plot(img: np.ndarray, title: str):
    plt.figure()
    plt.imshow(img, cmap = plt.cm.gray)
    plt.title(title)

# adds a black border around the image
def add_border(img: np.ndarray, border_size: int):
    height, width = img.shape
    border_img = np.zeros((height+2, width+2))

    for (x,y), value in np.ndenumerate(img):
        border_img[x+1][y+1] = value

    return border_img

def convolve_sobel(img : np.ndarray):
    sobel_kernel = [[1,  0, -1],
                    [-2, 0, -2],
                    [1,  0, -1]]
    
    res = scipy.signal.convolve2d(img, sobel_kernel, mode = 'same', boundary = 'fill', fillvalue = 0)
    return res

def main():
    # Create a black image
    img = np.zeros((640, 480))
    # ... and make a white rectangle in it
    img[100:-100, 80:-80] = 1

    # See how it looks
    # plot(img, 'Original')
    # plot(convolve_sobel(img), 'Sobel Convolution')

    sx = ndimage.sobel(img,axis=0,mode='constant')
    sy = ndimage.sobel(img,axis=1,mode='constant')

    # plot(sx, 'Sobel X')
    # plot(sy, 'Sobel Y')

    sobel = np.hypot(sx, sy)
    plot(sobel, 'SSD Sobel')
    plt.show()


if __name__ == "__main__":
    main()

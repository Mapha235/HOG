import scipy.ndimage as nd
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from PIL import Image


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

    sobel_kernel = np.array([[-1, 0, 1]])

    res = scipy.signal.convolve2d(img, sobel_kernel, mode='same')

    return res


def preprocessing(img):
    pass

def l2_norm():
    pass

def main():
    img = Image.open('./data/pedestrians.jpg')
    img = img.resize((np.array((1200, 600))))
    # plot(img, 'Orig')
    # img = plt.imread('./data/pedestrians.jpg')

    # plt.imshow(img)
    # img.astype('int32')
    # print(type(img))
    # Create a black image
    img = np.zeros((640, 480))
    # ... and make a white rectangle in it
    img[100:-100, 80:-80] = 1

    sx = nd.sobel(img, axis=0, mode='constant')
    sy = nd.sobel(img, axis=1, mode='constant')

    plot(sx, 'Sobel X')
    plot(sy, 'Sobel Y')

    # hypotenuse == root of squared sum == magnitude
    sobel = np.hypot(sx, sy)
    print(sobel.dtype)
    # sobel = nd.sobel(img,axis=1, mode ='constant')
    # plot(sx, 'sobel x')
    # plot(sy, 'Sobel y')
    # plot((sobel).astype(np.uint8), 'SSD')
    plot((convolve_sobel(img)), 'SSD')
    plt.show()


if __name__ == "__main__":
    main()

import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

def normalize_in_range(value, min, max):
    return (((value - 0) * (max - min)) / (100 - 0)) + min

def update_image(val):

    distortion = cv2.getTrackbarPos('Distortion', title_window)
    degrees = cv2.getTrackbarPos('Degrees', title_window)
    scale = cv2.getTrackbarPos('Scale', title_window)
    shear = cv2.getTrackbarPos('Shear', title_window)

    distortion = normalize_in_range(distortion, 0, 1)
    degrees = normalize_in_range(degrees, 0, 360)
    scale = normalize_in_range(scale, 1, 10)
    shear = normalize_in_range(shear, 0, 180)

    out = Image.fromarray(img)

    perspective = transforms.RandomPerspective(
        distortion_scale=distortion,
        p=1
    )
    affine = transforms.RandomAffine(
        degrees=degrees,
        scale=(scale, scale),
        shear=shear
    )

    out = perspective(out)
    out = affine(out)
    out = np.array(out)
    cv2.imshow(title_window, out)

if __name__ == '__main__':
    initial_distortion = 0
    initial_degrees = 0
    initial_scale = 1
    initial_shear = 0
    title_window = 'Torchvision Transform Tuning'

    img_file = 'papafles.jpg'
    img = cv2.imread(img_file)

    cv2.namedWindow(title_window)
    cv2.createTrackbar('Distortion', title_window, initial_distortion, 100, update_image)
    cv2.createTrackbar('Degrees', title_window, initial_degrees, 100, update_image)
    cv2.createTrackbar('Scale', title_window, initial_scale, 100, update_image)
    cv2.createTrackbar('Shear', title_window, initial_shear, 100, update_image)
    update_image(1)

    # Wait until user press some key
    cv2.waitKey()


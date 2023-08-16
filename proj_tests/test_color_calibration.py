"""
Script containing tests for the color calibration and color detection tools

The purpose of this script is to provide methods for testing the functionality of:

1) Detecting the color card in an image
2) Calibrating the color of an image given a reference image assuming the color card exists
"""

import cv2
import numpy as np

from util_tools.calibrate_colors import get_color_card, calibrate_image, calibrate_image_alt

DETECTRON_MODEL_PATH = "data/model_weights/palette_detector.pth"
EX_IMAGE_1 = "data/examples/butterfly_1.jpg"
EX_IMAGE_2 = "data/examples/butterfly_2.jpg"

def test_get_color_card():
    """
    Tests the detection of a color card. Utilizes the get_color_card function.
    """
    
    ex1 = cv2.imread(EX_IMAGE_1, cv2.IMREAD_COLOR)

    ex1_card = get_color_card(ex1, DETECTRON_MODEL_PATH)
    cv2.imwrite("tmp/palette.png", np.array(ex1_card, dtype=np.uint8))

def test_calibrate_images():
    """
    Tests the calibration of the color of one image using another as reference. This test
    utilizes the external model used in the calibrate_image function.
    """

    ex1 = cv2.imread(EX_IMAGE_1, cv2.IMREAD_COLOR)
    ex2 = cv2.imread(EX_IMAGE_2, cv2.IMREAD_COLOR)

    test = calibrate_image(ex1, ex2, DETECTRON_MODEL_PATH)
    cv2.imwrite("tmp/calibrated.png", np.array(test, dtype=np.uint8))

def test_calibrate_images_alt():
    """
    Tests the calibration of the color of one image using another as reference. This test
    utilizes the external model used in the calibrate_image_alt function.
    """

    ex1 = cv2.imread(EX_IMAGE_1, cv2.IMREAD_COLOR)
    ex2 = cv2.imread(EX_IMAGE_2, cv2.IMREAD_COLOR)

    test = calibrate_image_alt(ex1, ex2, DETECTRON_MODEL_PATH)
    cv2.imwrite("tmp/calibrated_alt.png", np.array(test, dtype=np.uint8))
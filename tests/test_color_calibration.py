import cv2
import numpy as np

from util_tools.calibrate_colors import get_color_card, calibrate_image

DETECTRON_MODEL_PATH = "data/model_weights/palette_detector.pth"
EX_IMAGE_1 = "data/examples/butterfly_1.jpg"
EX_IMAGE_2 = "data/examples/butterfly_2.jpg"

def test_get_color_card():
    ex1 = cv2.imread(EX_IMAGE_1, cv2.IMREAD_COLOR)

    ex1_card = get_color_card(ex1, DETECTRON_MODEL_PATH)
    cv2.imwrite("tmp/palette.png", np.array(ex1_card, dtype=np.uint8))

def test_calibrate_images():
    ex1 = cv2.imread(EX_IMAGE_1, cv2.IMREAD_COLOR)
    ex2 = cv2.imread(EX_IMAGE_2, cv2.IMREAD_COLOR)

    test = calibrate_image(ex1, ex2, DETECTRON_MODEL_PATH)
    cv2.imwrite("tmp/calibrated.png", np.array(test, dtype=np.uint8))
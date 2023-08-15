import cv2
import numpy as np

from tools.calibrate_colors import get_color_card

def test_get_color_card():
    ex1 = cv2.imread("data/examples/butterfly_1.jpg", cv2.IMREAD_COLOR)
    ex2 = cv2.imread("data/examples/butterfly_2.jpg", cv2.IMREAD_COLOR)

    ex1_card = get_color_card(ex1, "data/model_weights/palette_detector.pth")
    cv2.imwrite("tmp/palette.png", np.array(ex1_card, dtype=np.uint8))
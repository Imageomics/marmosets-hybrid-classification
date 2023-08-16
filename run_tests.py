"""
Test runner script for this project.

The purpose of this script is the run the tests in the proj_tests module. There
may be options to run sections of tests or individual tests.
"""

from tests.test_color_calibration import test_get_color_card, test_calibrate_images, test_calibrate_images_alt

if __name__ == "__main__":
    test_get_color_card()
    test_calibrate_images()
    test_calibrate_images_alt()
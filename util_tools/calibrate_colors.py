import cv2
import numpy as np

from externals.palette.Prediction.metadata_palette import setup_model, classes_dict, prediction_image
from externals.color_correction.color_correction import _match_cumulative_cdf_mod
from externals.color_calibration.src.color_calibration import match_histograms

def get_color_card(images, model_path):
    model = setup_model(model_path=model_path)

    is_input_list = type(images) is list
    if not is_input_list:
        images = [images]

    pred_imgs = []
    for image in images:
        output = model(image)
        instances = output['instances']
        fields = instances.get_fields()
        card_bbox = fields['pred_boxes'].tensor[0].cpu().numpy()
        pred_img = image[int(round(card_bbox[1])):int(round(card_bbox[3])),
            int(round(card_bbox[0])):int(round(card_bbox[2]))]
        pred_imgs.append(pred_img)

    if not is_input_list:
        return pred_imgs[0]
    
    return pred_imgs

def create_calibration_mapping(ref, tgt):
    color_map = np.zeros(256)
    k = 0
    for src_i in range(len(tgt)):
        for ref_i in range(len(ref)):
            if ref[ref_i] >= tgt[src_i]:
                k = ref_i
                break
        color_map[src_i] = k
    return color_map



def calibrate_image(ref_image, tgt_image, model_path):
    """
    Assumes the color cards are in the same orientation
    """

    [ref_card, tgt_card] = get_color_card([ref_image, tgt_image], model_path)

    tgt_card = cv2.resize(tgt_card, ref_card.shape[:2], interpolation= cv2.INTER_LINEAR)

    ref_channels = list(cv2.split(ref_card))
    tgt_channels = list(cv2.split(tgt_card))
    assert len(ref_channels) == len(tgt_channels), "# of reference image channels must match # of target image channels"
    full_tgt_channels = list(cv2.split(tgt_image))

    calibrated_channels = [ 
        _match_cumulative_cdf_mod(
            tgt_channels[i], 
            ref_channels[i], 
            full_tgt_channels[i]
        ) 
        for i in range(len(full_tgt_channels)) ]
    
    calibrated_image = cv2.merge(calibrated_channels)

    return calibrated_image

def calibrate_image_alt(ref_image, tgt_image, model_path):
    """
    Assumes the color cards are in the same orientation
    """

    [ref_card, tgt_card] = get_color_card([ref_image, tgt_image], model_path)

    calibrated_image = match_histograms(ref_card, tgt_card, tgt_image)

    return calibrated_image


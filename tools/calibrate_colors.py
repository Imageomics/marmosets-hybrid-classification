import cv2
import numpy as np

from externals.palette.Prediction.metadata_palette import setup_model, classes_dict, prediction_image

def get_color_card(image, model_path):
    model = setup_model(model_path=model_path)
    output = model(image)
    instances = output['instances']
    fields = instances.get_fields()
    card_bbox = fields['pred_boxes'].tensor[0].cpu().numpy()
    print(card_bbox)
    pred_img = image[int(round(card_bbox[1])):int(round(card_bbox[3])),
        int(round(card_bbox[0])):int(round(card_bbox[2]))]
    #pred_img = prediction_image(image, instances, classes_dict=classes_dict)

    return pred_img

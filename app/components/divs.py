from dash import html
from .query import get_sample

# Fixed styles
PRINT_STYLE = {'textAlign': 'center', 
               'color': 'MidnightBlue', 
               'font-size': '18px',
               'margin-bottom' : 20}
H4_STYLE = {'color': 'MidnightBlue', 
            'font-size': '20px',
            'margin-bottom' : 10,
            'margin-left': 5}
HALF_DIV_STYLE = {'height': '75%', 
                  'width': '48%', 
                  'display': 'inline-block'}
HR_STYLE = {'border-color': 'DarkOliveGreen',
            'borderWidth': '1px'}
ERROR_STYLE = {'textAlign': 'center', 
               'color': 'FireBrick', 
               'margin-bottom' : 10}

# URL for instructions
DOCS_URL = "https://github.com/Imageomics/marmosets-hybrid-classification/#app-how-it-works"

#Species dictionary for printing information
SPECIES_DICT = {"A": "Callithrix aurita", 
                    "AH": "Callithrix aurita hybrid", 
                    "J": "Callithrix jacchus", 
                    "P": "Callithrix penicillata", 
                    "PJ": "Callithrix penicillata x Callithrix jacchus hybrid"}

def get_error_div(error_dict):
    '''
    Function to return appropriate error message if there's a problem uploading the image.

    Parameters:
    -----------
    error_dict - Dictionary containing information about the error. Potential keys are 'type', 'unicode', and 'other'.

    Returns:
    --------
    error_div - Div with the corresponding error message.

    '''
    if 'type' in error_dict.keys():
        error_div = html.Div([
                            html.H4(["The source file is not a valid JPG or PNG format, please see the ",
                                     html.A("documentation", 
                                            href = DOCS_URL,
                                            target = '_blank',
                                            style = ERROR_STYLE),
                                     "."],
                            style = ERROR_STYLE)
        ])
    elif 'unicode' in error_dict.keys():
        error_div = html.Div([
            html.H4("There was a UnicodeDecode error processing this file.",
                    style = ERROR_STYLE)
        ])
    else:
        error_div = html.Div([
            html.H4("There was an error processing this file.",
                    style = ERROR_STYLE)
        ])
    return error_div

def get_results_div(pred_species, confidence, img_src):
    '''
    Function to return div with prediction, confidence, uploaded image, and sample image of predicted species.

    Parameters:
    -----------
    pred_species - String. Species predicted by model ("A", "AH", "J", "P", "PJ"), key to SPECIES_DICT for full name.
    confidence - Float. Confidence of the prediction (percentage).
    img_src - Source for uploaded image to display.

    Returns:
    --------
    results_div - Div with the corresponding results of the model and recommendations.

    '''
    # Get sample image and native region for predicted species
    sample = get_sample(pred_species)

    results_div = [html.H4(f"The uploaded image is likely a picture of {SPECIES_DICT[pred_species]}, with confidence {confidence}%.",
                        style = PRINT_STYLE),
                html.Hr(style = HR_STYLE),
                html.Div([html.Div([html.H4("Uploaded Image: ",
                                  style = H4_STYLE),
                            html.Img(src = img_src,
                                     style = {'height': '98%',
                                              'width': '98%'})],
                            style = HALF_DIV_STYLE),
                        html.Div([html.H4(f"Sample Image of {SPECIES_DICT[pred_species]}: ",
                                  style = H4_STYLE),
                            #html.Img(src = sample['img_path'])],
                            # dummy data doesn't have images
                            html.H4(sample['image'])],
                            style = HALF_DIV_STYLE)]),
                html.Br(),
                html.Hr(style = HR_STYLE),
                html.H4(f"Note that this species is {sample['native_region']}.",
                                  style = PRINT_STYLE)]
    return results_div

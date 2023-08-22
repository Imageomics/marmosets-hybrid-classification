import base64
import io
import json
import numpy as np
from PIL import Image
from infer import infer
from dash import Dash, html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
from app.components.divs import  get_error_div, get_results_div

# Fixed style
H1_STYLE = {'textAlign': 'center', 
            'color': 'MidnightBlue'
            }
UPLOAD_STYLE = {'color': 'MidnightBlue', 
                'border-color': 'DarkOliveGreen',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'font-size': '18px',
                'font-weight': 'bold',
                'textAlign': 'center',
                'lineHeight': '70px',
                'width': '100%',
                'height': '70px'
                }
HR_STYLE = {'border-color': 'DarkOliveGreen',
            'borderWidth': '1px'}

# Model weights
MODEL_WEIGHTS = "app/data/marmoset_classifier.pt"

# Initialize app and set layout
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
                html.H1("Marmoset Classifier", style = H1_STYLE),
                dcc.Upload(id = 'upload-img',
                           children = html.Div([
                                        'Drag and Drop Image (JPG or PNG) or ',
                                        html.A('Select file')
                                ],
                                        style = UPLOAD_STYLE
                                                ),
                            multiple = False
                            ),
        
                # Set up memory store with loading indicator, will revert on page refresh
                dcc.Loading(id = 'memory-loading',
                            type = "circle",
                            color = 'DarkMagenta',
                            children = dcc.Store(id = 'memory')),
                html.Hr(style = HR_STYLE),
                
                dcc.Loading(id = 'output-img-upload-loading',
                            type = "circle",
                            color = 'DarkMagenta',
                            children = html.Div(
                                        id = 'output-img-upload'))
])

# Image read in and saved to memory
@app.callback(
        Output('memory', 'data', allow_duplicate=True),
        Input('upload-data', 'contents'),
        State('upload-data', 'filename'),
        prevent_initial_call = True
)

def parse_contents(contents, filename):
    '''
    Function to read uploaded image and save to PIL format.
    '''
    if contents is None:
        raise PreventUpdate
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'jpg' or 'jpeg' or 'png' in filename.lower():
            im = Image.open(io.BytesIO(decoded))
        else:
            return json.dumps({'error': {'type': 'wrong file type'}})
    except UnicodeDecodeError as e:
        print(e)
        return json.dumps({'error': {'unicode': str(e)}})
    
    except Exception as e:
        print(e)
        return json.dumps({'error': {'other': str(e)}})
    
    im_list = np.array(im).tolist()

    data = {'pil_img': im_list,
            'img_src': contents}

    return json.dumps(data)

# Callback to update data if new image uploaded
@app.callback(
        Output('memory', 'data'),
        Input('upload-img', 'contents'),
        State('upload-img', 'filename'),
        prevent_initial_call = True
)
    
def update_output(contents, filename):
    if contents is not None:
        return parse_contents(contents, filename)

# Callback to get results div (prediction, confidence, sample)
@app.callback(
        Output('output-img-upload', 'children'),
        Input('memory', 'data'),
        prevent_initial_call = True
)

def get_display(jsonified_data):
    '''
    Function to call model on uploaded image and return prediction, confidence, and sample image alongside uploaded image.
    Returns error div if error occurs in upload.
    '''
    # load saved data
    data = json.loads(jsonified_data)
    if 'error' in data:
        return get_error_div(data['error'])
    
    # Convert list back to PIL image
    pil_img = np.asarray(data['pil_img'], dtype = np.uint8)
    pil_img = Image.fromarray(pil_img)
     
    species_labels = ["A", "AH", "J", "P", "PJ"]
    
    # Get prediction and confidence from model
    prediction_dict = infer(pil_img.convert('RGB'), MODEL_WEIGHTS)
    pred_idx = prediction_dict['prediction']
    pred_species = species_labels[pred_idx]
    confidences = prediction_dict['confidences']
    confidence = confidences[pred_idx]*100

    # Get div with prediction, confidence, and uploaded and sample images
    children = get_results_div(pred_species, np.round(confidence, 2), data['img_src'])

    return children

if __name__ == '__main__':
    app.run()

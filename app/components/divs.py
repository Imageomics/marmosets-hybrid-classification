from dash import html, dcc

# Fixed styles and sorting options
H1_STYLE = {'textAlign': 'center', 'color': 'MidnightBlue'}
H4_STYLE = {'color': 'MidnightBlue', 'margin-bottom' : 10}
HALF_DIV_STYLE = {'height': '48%', 'width': '48%', 'display': 'inline-block'}
QUARTER_DIV_STYLE = {'width': '24%', 'display': 'inline-block'}
BUTTON_STYLE = {'color': 'MidnightBlue', 
                'background-color': 'BlanchedAlmond', 
                'border-color': 'MidnightBlue',
                'font-size': '15px'}
ERROR_STYLE = {'textAlign': 'center', 'color': 'FireBrick', 'margin-bottom' : 10}
DOCS_URL = "https://github.com/Imageomics/marmosets-hybrid-classification/tree/main/app/README.md"

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

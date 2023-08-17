import pandas as pd
from dash import html

# Source csv for image information
SOURCE_CSV = "data/marmoset_imgs(placeholder).csv"
df = pd.read_csv(SOURCE_CSV)

def get_prediction(pil_img, model_weights):
    '''
    Function to call the trained model on the uploaded image to get predicted species and confidence of prediction.
    
    Parameters:
    -----------
    pil_img - Uploaded image in Pil format.
    model_weigths - Weights for trained model.
    
    Returns:
    --------
    pred_dict - Dictionary with predicted species ("prediction") and "confidence" of the prediction.
    '''
    # Call model on given image
    # pred_dict = model_call(pil_img, model_weights)
    
    # dummy data for testing app
    pred_dict = {'prediction': 'A',
                 'confidence': [80]}
    
    return pred_dict

# Retrieve sample image and native region of predicted species

def get_sample(pred_species):
    '''
    Function to retrieve a sample image of the predicted species and its native region.

    Parameters:
    -----------
    pred_species - String. Predicted species of the uploaded image.

    Returns:
    --------
    results - Dictionary with html image element with `src` element pointing to path for the sample image of the predicted species ('img_path')
                and its native location ('native_region').
    '''
    # Separate functions in case we have more options later
    filepath, native_region = get_sample_data(pred_species)
    
    results = {}
    # Currently waiting on public URLs, so filepath is just a name
    #results['img_path'] = html.Img(src = filepath)
    results['img_path'] = filepath
    if native_region == 'Exotic':
        region = "exotic. It should not be released in the wild"
    else:
        region = "native to " + native_region + ". Perform genetic testing to ensure it isn't a cryptic hybrid before releasing it in " + native_region
    results['native_region'] = region

    return results

def get_sample_data(pred_species):
    '''
    Funtion to randomly select a sample image of the given species and provide associated filepath and native location.
    
    Parameters:
    -----------
    pred_species - String. Predicted species of the uploaded image.

    Returns:
    --------
    filepath - Filepath (URL) corresponding to the sample image of predicted species. 
    native_region - Native region of the species, "Exotic" if hybrid.
    
    '''
    df_sub = df.loc[df.Species == pred_species].copy()
    df_filtered = df_sub.sample()
    filepath = df_filtered.img_url.astype('string').values
    native_region = df_filtered.native_region.astype('string').values
    #return filepath for randomly selected image from the filtered dataset and native region of species
    return filepath, native_region

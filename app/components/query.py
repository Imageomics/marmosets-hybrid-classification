import pandas as pd

# Source csv for image information
SOURCE_CSV = "app/data/marmoset_imgs(placeholder).csv"
df = pd.read_csv(SOURCE_CSV)

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
    results['image'] = filepath
    if native_region == 'Exotic':
        region = "exotic. It should not be released in the wild"
    else:
        region = f"native to {native_region}, but perform genetic testing to ensure it isn't a cryptic hybrid before releasing it in {native_region}"
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
    return filepath[0], native_region[0]

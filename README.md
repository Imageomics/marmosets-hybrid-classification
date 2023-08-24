# Hybrid Classification of Brazilian Marmosets

Hybrid marmosets are being released in areas causing damage to the native marmosets population in the local ecosystem. Identification of these hybrids is vital to protecting the local endangered species of marmosets. We aim to train state of the art image classification models to classify hybrid marmosets from parent marmoset species. The project involves image annotation, color standardization, model training, and mobile app creation.


## How it Works

### Installation

First install required packages: 
```
pip install -r requirements.txt
```
dectectron2 must be installed separately after:
```
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**Note:** If running on Windows, may need to install Windows C++ build tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/.
This is needed to install [detectron2](https://github.com/facebookresearch/detectron2/issues/4948).

### Data Gathering and Model Training

The images used to train this model are compiled in the [hybrid_photos dataset](https://huggingface.co/datasets/callithrix/hybrid_photos) on Hugging Face. To access the data, run  
```
python download.py --dest_path <filepath> --repo_id <repoID>
```
Then, to generate the model, run 
```
python train.py
``` 

This will run the appropriate scripts to process the data for training and train the model.

## Testing

Testing can be done with `run_tests.py`, though some paths are hard-coded.


<hr>



# Hybrid Classification of Brazilian Marmosets App

The app takes in an image of a marmoset and runs it through our classifier to determine its species (the confidence of the prediction is also provided). 

Additionally, the app displays the uploaded image alongside a sample image of the predicted species. It informs the user of the predicted species' natural habitat or that it is "exotic", meaning that it is a hybrid, and therefore has no natural habitat.


## App: How it Works

There is an actively running instance of the marmoset classifier app available [here](https://huggingface.co/spaces/callithrix/marmoset-classifier). The user simply uploads a `JPG` or `PNG` image, and the results will be displayed below. Note that all information is session-specific, so refreshing the page will clear the app cache.

If you want to run your own instance of the app, follow the instructions below.

### App Installation

First install required packages: 
```
pip install -r app/requirements.txt
```

**Note:** The requirements for the app are not the same as the requirements for the model.

### Sample Images

The sample images come from our training set, and are compiled in the [hybrid_photos dataset](https://huggingface.co/datasets/callithrix/hybrid_photos) on Hugging Face. 


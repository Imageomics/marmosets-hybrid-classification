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

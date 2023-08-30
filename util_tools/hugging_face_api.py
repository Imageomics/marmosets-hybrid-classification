"""
To utilize and access Huggingface features

The purpose of this script is to provide functionality to download and access
Huggingface resources.
"""

from huggingface_hub import login, snapshot_download

def download_dataset(repo_id, local_dest="data/huggingface_dset", user_access_token=None, ignore_NEF=False):
    """
    Args:
        repo_id (str): Hugging face repository id
        local_dest (str): Path to directory will files or symlinks to files will be downloaded to
        user_access_token (str): Huggingface user access token. May be needed if dataset is private
        ignore_NEF (bool): ignore larget .NEF files when downloading

    Description:
        This function will download a snapshot of a Huggingface dataset given the repo id. The user
        access token may be needed if accessing a private dataset that requires authentication.
    """

    ignore_patterns = []
    if ignore_NEF:
        ignore_patterns.append("*.NEF")
    
    login(user_access_token)
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dest, 
        local_dir_use_symlinks=False, ignore_patterns=ignore_patterns)
"""
Script for downloading the marmoset dataset.

The purpose of this script is download the marmoset dataset from Huggingface.
"""

from argparse import ArgumentParser
from util_tools.hugging_face_api import download_dataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--user_access_token", type=str, default=None)
    parser.add_argument("--repo_id", type=str, default=None, required=True)
    parser.add_argument("--dest_path", type=str, default="data/marmosets")

    args = parser.parse_args()

    download_dataset(args.repo_id, local_dest=args.dest_path, user_access_token=args.user_access_token)


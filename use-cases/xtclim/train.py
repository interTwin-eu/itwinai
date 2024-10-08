"""
Train file to launch pipeline
"""
import os
import sys
from typing import Dict
import argparse
import logging
from datetime import datetime
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'preprocessing'))

from itwinai.parser import ConfigParser

def read_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":

    config = read_config('pipeline.yaml')
    seasons_list = config['seasons']
    for season in seasons_list:
        config['pipeline']['init_args']['steps']['training-step']['init_args']['seasons'] = season
        model_uri = f"outputs/cvae_model_{season}1d_1memb.pth"
        config['pipeline']['init_args']['steps']['evaluation-step']['init_args']['model_uri'] = model_uri
        config['pipeline']['init_args']['steps']['evaluation-step']['init_args']['seasons'] = season
        pipe_parser = ConfigParser(
            config=config,
        )
        pipeline = pipe_parser.parse_pipeline()
        print(f"Running pipeline for season: {season}")
        pipeline.execute()


"""
Train file to launch pipeline
"""

import os
import sys
from itwinai.parser import ConfigParser
from itwinai.utils import load_yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "preprocessing"))


if __name__ == "__main__":

    config = load_yaml('pipeline.yaml')
    seasons_list = config['seasons']

    for season in seasons_list:
        model_uri = f"outputs/cvae_model_{season}1d_1memb.pth"
        override_dict = {
            'season': season,
            'model_uri': model_uri
        }
        pipe_parser = ConfigParser(
            config=config,
            override_keys=override_dict
        )
        pipeline = pipe_parser.parse_pipeline()

        print(f"Running pipeline for season: {season}")
        pipeline.execute()
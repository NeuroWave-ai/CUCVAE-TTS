import argparse

import yaml

from preprocessor.mutil_preprocessor import MutilPreprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    parser.add_argument("--index", type=int, help="int index")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessor = MutilPreprocessor(config)
    preprocessor.build_from_path(args.index)

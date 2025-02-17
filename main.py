"""
This file will serve as the main entry point for the application.
"""
import argparse

from src.polaris import PolarisDispatcher
from src.utils import load_yaml_to_dict


def main(args: dict) -> None:
    config_filename = args["config_filename"]
    params: dict = load_yaml_to_dict(config_filename)
    print(params)

    polaris_dispatcher = PolarisDispatcher(params)
    polaris_dispatcher.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pass in the config file.")
    parser.add_argument("--config_filename", help="Name of the config file in the config directory.")

    input_args = parser.parse_args()
    input_args_dict = vars(input_args)
    main(input_args_dict)

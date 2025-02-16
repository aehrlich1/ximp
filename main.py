"""
This file will serve as the main entry point for the application.
"""
import argparse

from src.utils import load_yaml_to_dict
from src.polaris import Polaris


def main(args: dict) -> None:
    config_filename = args["config_filename"]
    params: dict = load_yaml_to_dict(config_filename)
    print(params)

    match params["task"]:
        case "antiviral-ligand-poses":
            raise NotImplementedError
        case "antiviral-potency":
            polaris = Polaris(params)
            polaris.run()
        case "antiviral-admet":
            raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pass in the config file.")
    parser.add_argument("--config_filename", help="Name of the config file in the config directory.")

    input_args = parser.parse_args()
    input_args_dict = vars(input_args)
    main(input_args_dict)

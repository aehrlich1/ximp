"""
This file will serve as the main entry point for the application.
"""
import argparse

from src.utils import load_yaml_to_dict
from src.admet_potency import main as admet_main


def main(args: dict) -> None:
    config_filename = args["config_filename"]
    params: dict = load_yaml_to_dict(config_filename)
    print(params)

    match params["task"]:
        case "antiviral-ligand-poses":
            print("Antiviral Ligand Poses is not yet implemented.")
        case "antiviral-potency":
            admet_main(params)
        case "antiviral-admet":
            admet_main(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pass in the config file.")
    parser.add_argument("--config_filename", help="Name of the config file in the config directory.")

    input_args = parser.parse_args()
    input_args_dict = vars(input_args)
    main(input_args_dict)

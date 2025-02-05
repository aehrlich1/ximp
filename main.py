"""
This file will serve as the main entry point for the application.
"""
import argparse


def main():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pass in the config file.")
    parser.add_argument("--config_filename", help="Name of the config file in the config directory.")

    input_args = parser.parse_args()
    main()

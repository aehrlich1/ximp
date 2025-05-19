"""
This file will serve as the main entry point for the application.
"""

import argparse

from src.polaris import Polaris


def main(params: dict) -> None:
    polaris = Polaris(params=params)
    polaris.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass in the parameters.")

    # Task parameters
    parser.add_argument("--task", help="Task name", default="potency")
    parser.add_argument("--target_task", help="Target Task", default="pIC50 (MERS-CoV Mpro)")

    # Learning parameters
    parser.add_argument("--batch_size", help="Batch size", default=32)
    parser.add_argument("--dropout", help="Dropout ratio", default=0.0)
    parser.add_argument("--epochs", help="Epochs", default=100)
    parser.add_argument("--lr", help="Learning Rate", default=1.0e-4)
    parser.add_argument("--weight_decay", help="Weight decay", default=0)

    # Model parameters
    parser.add_argument("--repr_model", help="Representation Model", default="GIN")
    parser.add_argument("--num_layers", help="Number of GNN layers", default=3)
    parser.add_argument("--encoding_dim", help="Encoding dimension of features", default=8)
    parser.add_argument("--hidden_channels", help="Number of hidden channels", default=32)
    parser.add_argument("--out_channels", help="Number of output channels", default=64)
    parser.add_argument("--proj_hidden_dim", help="Projection hidden dimension", default=64)
    parser.add_argument("--out_dim", help="Output dimension", default=1)

    # Dataset parameters
    parser.add_argument("--num_cv_folds", help="Number of cross-validation folds", default=5)
    parser.add_argument("--num_cv_bins", help="Number of cross-validation bins", default=10)
    parser.add_argument(
        "--scaffold_split_val_sz", help="Scaffold spit validation slide", default=0.1
    )

    # EHIMP Paramters
    parser.add_argument("--use_erg", help="Use ERG", default=False)
    parser.add_argument("--use_ft", help="Use Feature Tree", default=False)
    parser.add_argument("--ft_resolution", help="Feature tree resolution", default=1)
    parser.add_argument("--rg_embedding_dim", help="Reduced graph embedding dimention", default=8)

    input_args = parser.parse_args()
    input_args_dict = vars(input_args)
    main(input_args_dict)

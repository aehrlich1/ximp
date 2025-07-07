"""
This file will serve as the main entry point for the application.
"""

import argparse

from src.trainer import Trainer
from src.utils import str2bool


def main(params: dict) -> None:
    trainer = Trainer(params=params)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass in the parameters.")

    # Task parameters
    parser.add_argument("--task", help="Task name", default="potency")
    parser.add_argument("--target_task", help="Target Task", default="pIC50 (MERS-CoV Mpro)")

    # Learning parameters
    parser.add_argument("--batch_size", help="Batch size", default=32, type=int)
    parser.add_argument("--dropout", help="Dropout ratio", default=0.0, type=float)
    parser.add_argument("--epochs", help="Epochs", default=100, type=int)
    parser.add_argument("--lr", help="Learning Rate", default=1.0e-4, type=float)
    parser.add_argument("--weight_decay", help="Weight decay", default=0, type=float)

    # Model parameters
    parser.add_argument("--repr_model", help="Representation Model", default="GIN")
    parser.add_argument("--num_layers", help="Number of GNN layers", default=3, type=int)
    parser.add_argument(
        "--encoding_dim", help="Encoding dimension of features", default=8, type=int
    )
    parser.add_argument("--hidden_channels", help="Number of hidden channels", default=32, type=int)
    parser.add_argument("--out_channels", help="Number of output channels", default=64, type=int)
    parser.add_argument(
        "--proj_hidden_dim", help="Projection hidden dimension", default=64, type=int
    )
    parser.add_argument("--out_dim", help="Output dimension", default=1, type=int)

    # Dataset parameters
    parser.add_argument(
        "--num_cv_folds", help="Number of cross-validation folds", default=5, type=int
    )
    parser.add_argument(
        "--num_cv_bins", help="Number of cross-validation bins", default=10, type=int
    )
    parser.add_argument(
        "--scaffold_split_val_sz", help="Scaffold spit validation slide", default=0.1, type=float
    )

    # EHIMP Paramters
    parser.add_argument(
        "--use_erg", help="Use ERG", default=False, type=str2bool, const=True, nargs="?"
    )
    parser.add_argument(
        "--use_ft", help="Use Feature Tree", default=False, type=str2bool, const=True, nargs="?"
    )
    parser.add_argument("--ft_resolution", help="Feature tree resolution", default=1, type=int)
    parser.add_argument(
        "--rg_embedding_dim", help="Reduced graph embedding dimention", default=8, type=int
    )

    input_args = parser.parse_args()
    input_args_dict = vars(input_args)
    main(input_args_dict)

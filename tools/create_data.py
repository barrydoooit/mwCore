import argparse
from pathlib import Path
from typing import Any, List, Literal
import sys
import os

import debugpy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import traceback



def mri_data_prep(root_path: str,
                  out_dir: str,
                  split: Literal[1, 2] = 2,
                  protocol: Literal[1, 2] = 2,
                  ratio: float = 0.8,
                  seed: int = 42,
                  ):
    from tools.dataset_converters.mri_converter import MRIDatasetConverter
    converter = MRIDatasetConverter(
        input_root=Path(root_path) / 'dataset_release' / 'aligned_data',
        output_root=Path(out_dir),
        split=split,
        protocol=protocol,
        ratio=ratio,
        seed=seed)
    converter.process_all()

def ask_for_option(prompt: str, options: List[str], default: Any = None) -> str:
    option = input(prompt).strip().lower()
    if default is not None and option == '':
        return default
    while option not in options:
        print(f"Invalid option. Please choose from {options}")
        option = input(prompt).strip().lower()
    return option

def main():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument('dataset', help='name of the dataset')
    parser.add_argument(
        '--root-path',
        type=str,
        help='specify the root path of the dataset')
    parser.add_argument(
        '--out-dir',
        type=str,
        help='specify the output directory')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    args = parser.parse_args()
    if args.debug:
        debugpy.listen(5678)
        print('Waiting for debugger attach')
        debugpy.wait_for_client()
    if args.dataset == 'mri':
        use_defaults = ask_for_option(
            prompt="Use default settings (split=S2, protocol=P2, ratio=0.8, seed=42) (y/n)? ",
            options=['y', 'n']
        )
        if use_defaults == 'y':
            split, protocol, ratio, seed = 2, 2, 0.8, 42
        else:
            split = int(ask_for_option(
                prompt=(
                    "Choose data-split setting (default 2):\n"
                    "  1 (S1 Random Split) - random 80%/20% split of all samples\n"
                    "  2 (S2 Split by Subjects) - train on 80% of subjects, test on the rest\n"
                    "Enter 1 or 2: "
                ),
                options=['1', '2'],
                default='2'
            ))
            protocol = int(ask_for_option(
                prompt=(
                    "Choose evaluation protoco (default 3l:\n"
                    "  1 (P1) - all 12 movements (stretching, relaxing in free form, and walking)\n"
                    "  2 (P2) - only the first 10 rehabilitation movements\n"
                    "  3 (P3) - only walking\n"
                    "Enter 1, 2, or 3: "
                ),
                options=['1', '2', '3'],
                default='3'
            ))
            ratio = float(input("Enter the train/val ratio (default: 0.8): ") or 0.8
            )
            seed = int(input("Enter the random seed (default: 42): ") or 42)

        mri_data_prep(
            root_path=args.root_path,
            out_dir=args.out_dir,
            split=split,
            protocol=protocol,
            ratio=ratio,
            seed=seed
        )
    elif args.dataset == 'paw':
        from tools.dataset_converters.paw_converter import AsteriosPawDatasetConverter
        converter = AsteriosPawDatasetConverter(
            input_root=args.root_path,
            output_root=args.out_dir,
            seed=42)
        converter.process_all()
        
if __name__ == '__main__':
    main()
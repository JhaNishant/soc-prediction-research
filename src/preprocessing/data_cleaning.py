import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Clean the dataset by dropping specified columns."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input dataset file.")
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output file for the cleaned dataset."
    )

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    # Load the dataset
    df = pd.read_csv(input_file)

    # Drop the specified columns
    df.drop(["sample_id", "geom_id", "date_id", "depth_id",
             "top_depth", "depth_len"], axis=1, inplace=True, )

    # Save the cleaned dataset to a file
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()

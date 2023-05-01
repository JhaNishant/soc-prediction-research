import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Calculate the correlation coefficients between "
                    "'soc_percent' and other features.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input dataset file.")
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output file for the correlation coefficients.",
    )

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    # Load the dataset
    df = pd.read_csv(input_file)

    # Calculate the correlation matrix
    corr_matrix = df.corr(method="pearson")

    # Get the correlation coefficients between 'soc_percent' and other features
    soc_percent_corr = corr_matrix["soc_percent"]

    # Save the result to a file
    soc_percent_corr.to_csv(output_file, index=True)


if __name__ == "__main__":
    main()

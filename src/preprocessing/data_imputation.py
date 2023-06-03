import argparse

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


def main():
    parser = argparse.ArgumentParser(
        description="Impute missing values in the dataset."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input dataset file."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output file for the dataset after imputation."
    )

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    # Load the dataset
    df = pd.read_csv(input_file)

    # Create an imputer
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=10, random_state=0))

    # List of column names
    cols = df.columns

    # Create a copy of the dataframe
    df_imputed = df.copy()

    for col in cols:
        # If the column has missing values
        if df[col].isnull().sum() > 0:
            # Print a message indicating that imputation is starting
            print(f"Imputing {col}...")
            # Perform imputation on the column and store the result in the copy
            df_imputed[col] = imputer.fit_transform(df[[col]])
            # Print a message indicating that imputation is complete
            print(f"Finished imputing {col}.")

    # Print a message indicating that all imputation is complete
    print("Imputation complete.")

    # The result is a numpy array, so convert it back to a DataFrame
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

    # Save the cleaned dataset to a file
    df_imputed.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()

import argparse
import faulthandler
import logging

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

faulthandler.enable()


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('Starting script...')

    parser = argparse.ArgumentParser(
        description="Perform KMeans clustering on the dataset and output "
                    "'soc_percent' values for each cluster.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input dataset file.")
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to the output folder where the 'soc_percent' "
             "values for each cluster will be saved.",
    )

    args = parser.parse_args()

    input_file = args.input_file
    output_folder = args.output_folder

    # Load the dataset
    df = pd.read_csv(input_file)

    # Normalize the features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(
            df.drop(
                'soc_percent',
                axis=1)),
        columns=df.columns.drop('soc_percent'))

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=0)
    clusters = kmeans.fit(df_scaled)

    # Add cluster labels to the DataFrame
    df['cluster'] = clusters.labels_

    # Output 'soc_percent' values for each cluster
    for cluster in set(df['cluster']):
        output_file = f"{output_folder}/cluster_{cluster}_soc_percent.csv"
        df[df['cluster'] == cluster]['soc_percent'].to_csv(
            output_file, index=False)
        logging.info(
            f"Saved 'soc_percent' values for cluster "
            f"{cluster} to {output_file}")

    logging.info('Finished script.')


if __name__ == "__main__":
    main()

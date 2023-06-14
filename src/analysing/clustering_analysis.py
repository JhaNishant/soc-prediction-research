import argparse
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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

    os.makedirs(output_folder, exist_ok=True)

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
    kmeans = KMeans(n_clusters=5, n_init=10, random_state=0)
    clusters = kmeans.fit(df_scaled)

    # Add cluster labels to the DataFrame
    df['cluster'] = clusters.labels_

    # Output 'soc_percent' values for each cluster
    for cluster in set(df['cluster']):
        output_file = os.path.join(
            output_folder, f"cluster_{cluster}_soc_percent.csv")
        df[df['cluster'] == cluster]['soc_percent'].to_csv(
            output_file, index=False)
        logging.info(
            f"Saved 'soc_percent' values for cluster "
            f"{cluster} to {output_file}")

    # Save the DataFrame with the cluster assignments to a CSV file
    df.to_csv(os.path.join(output_folder, "clustered_data.csv"), index=False)
    logging.info(
        f"Saved the DataFrame with the cluster assignments to "
        f"{os.path.join(output_folder, 'clustered_data.csv')}")

    # Set the style of the plots
    sns.set(style="whitegrid")

    # 1. Histograms of `soc_percent` values for each cluster
    for cluster in set(df['cluster']):
        plt.figure(figsize=(10, 6))
        plt.hist(df[df['cluster'] == cluster]
                 ['soc_percent'], bins=20, edgecolor='black')
        plt.title(f'Histogram of `soc_percent` for Cluster {cluster}')
        plt.xlabel('soc_percent')
        plt.ylabel('Frequency')
        plt.savefig(
            os.path.join(
                output_folder,
                f'histogram_cluster_{cluster}.png'))
        plt.close()

    # 2. Violin plots of `soc_percent` values for each cluster
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='cluster', y='soc_percent', data=df)
    plt.title('Violinplot of `soc_percent` by Cluster')
    plt.savefig(os.path.join(output_folder, 'violin_plot.png'))
    plt.close()

    # 3. Bar plot of the number of data points in each cluster
    plt.figure(figsize=(10, 6))
    sns.countplot(x='cluster', data=df, edgecolor='black')
    plt.title('Number of Data Points in Each Cluster')
    plt.savefig(os.path.join(output_folder, 'bar_plot.png'))
    plt.close()

    logging.info('Finished script.')


if __name__ == "__main__":
    main()

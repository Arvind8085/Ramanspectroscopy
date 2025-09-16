import pandas as pd

class ClusterStatisticsProcessor:
    def __init__(self, input_file, sheet_name):
        """
        Initialize the processor with the input file and sheet name to load data.
        """
        self.input_file = input_file
        self.sheet_name = sheet_name
        self.data = None
    
    def load_data(self):
        """
        Load the cluster statistics data from the specified sheet in the input file.
        """
        self.data = pd.read_excel(self.input_file, sheet_name=self.sheet_name)
    
    @staticmethod
    def calculate_group_mean(df, column_name):
        """
        Replace the original column with the mean of the column grouped by 'Cluster'.
        
        :param df: DataFrame to process
        :param column_name: Name of the column to replace with group means
        :return: Processed DataFrame
        """
        # Group by 'Cluster' and calculate the mean of the specified column for each cluster
        group_mean = df.groupby('Cluster')[column_name].mean().reset_index()
        
        # Merge the mean column back to the original dataframe
        df = pd.merge(df, group_mean, on='Cluster', suffixes=('', '_mean'))
        
        # Drop the original column
        df.drop(columns=column_name, inplace=True)
        
        # Rename the mean column
        df.rename(columns={f'{column_name}_mean': column_name}, inplace=True)
        
        return df
    
    def process_column(self, column_name):
        """
        Process a specified column by replacing it with its group mean.
        
        :param column_name: Name of the column to process
        """
        if self.data is not None:
            self.data = self.calculate_group_mean(self.data, column_name)
        else:
            raise ValueError("Data not loaded. Use load_data() before processing.")
    
    def save_to_file(self, output_file):
        """
        Save the processed DataFrame to an Excel file.
        
        :param output_file: Name of the output Excel file
        """
        if self.data is not None:
            self.data.to_excel(output_file, index=False)
        else:
            raise ValueError("Data not loaded. Use load_data() before saving.")

# Usage
if __name__ == "__main__":
    input_file = 'peak_statistics_R4.xlsx'
    sheet_name = 'Dataset 1 Stats'
    output_file = 'cluster_stats_processed.xlsx'
    
    # Initialize the processor
    processor = ClusterStatisticsProcessor(input_file, sheet_name)
    
    # Load data
    processor.load_data()
    
    # Process specified columns
    processor.process_column('mean_wavenumber')
    
    # Save the processed data to a file
    processor.save_to_file(output_file)
    
    print(f'Modified data has been saved to {output_file}')

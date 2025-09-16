import pandas as pd

class ExcelSheetProcessor3:
    def __init__(self, input_file):
        self.input_file = input_file
        self.sheet1_df = None
        self.sheet4_df = None
    
    def load_sheets(self):
        """Load Sheet1 and Sheet4 into dataframes."""
        self.sheet1_df = pd.read_excel(self.input_file, sheet_name='Sheet1')
        self.sheet4_df = pd.read_excel(self.input_file, sheet_name='Sheet4')
    
    def filter_columns(self):
        """Filter columns in Sheet1 based on the presence of intensity/sample names from Sheet4."""
        # Get the list of intensity/sample names from Sheet4
        intensity_names = self.sheet4_df.iloc[:, 1].tolist()

        # Preserve the first column of Sheet1 and identify other columns to keep
        columns_to_keep = [col for col in self.sheet1_df.columns[1:] if self.sheet1_df[col].isin(intensity_names).any()]  # Exclude first column
        columns_to_keep = [self.sheet1_df.columns[0]] + columns_to_keep  # Add the first column back

        # Filter Sheet1 to keep only the necessary columns
        sheet1_filtered = self.sheet1_df[columns_to_keep]
        
        return sheet1_filtered
    
    def save_to_excel(self, sheet1_filtered):
        """Save the modified Sheet1 dataframe back to the same Excel file."""
        # Save the modified Sheet1 back to the same Excel file
        with pd.ExcelWriter(self.input_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            # Overwrite Sheet1 with the modified data
            sheet1_filtered.to_excel(writer, sheet_name='Sheet1', index=False)
    
    def run(self):
        """Run the entire processing sequence."""
        self.load_sheets()
        sheet1_filtered = self.filter_columns()
        self.save_to_excel(sheet1_filtered)
        
class EnhancedExcelSheetProcessor(ExcelSheetProcessor3):
    def run(self):
        # Run original processing (load, filter, save)
        super().run()
        
        # Reload the updated Sheet1 from the file
        df_sheet1 = pd.read_excel(self.input_file, sheet_name='Sheet1')
        
        # Step 1: Replace header with transposed Sheet4 Column A values
        new_header = self.sheet4_df.iloc[:, 0].tolist()  # Get Column A values
        num_cols = len(df_sheet1.columns)
        # Ensure header matches number of columns
        new_header_adjusted = new_header[:num_cols] if len(new_header) >= num_cols else new_header + [None] * (num_cols - len(new_header))
        df_sheet1.columns = new_header_adjusted
        
        # Step 2: Copy value from A5 (index=3) to A1 (first column header)
        value_at_A5 = df_sheet1.iloc[3, 0]
        new_columns = list(df_sheet1.columns)
        new_columns[0] = value_at_A5
        df_sheet1.columns = new_columns
        
        # Step 3: Delete Row 5 (index=3)
        df_sheet1 = df_sheet1.drop(3).reset_index(drop=True)
        
        # Save the final modified Sheet1
        with pd.ExcelWriter(self.input_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df_sheet1.to_excel(writer, sheet_name='Sheet1', index=False)



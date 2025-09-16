import pandas as pd

class ExcelConcentrationMapper:
    def __init__(self, file_path, error_widget=None):
        self.file_path = file_path
        self.sheet1_df = None
        self.sheet2_df = None
        self.sheet4_df = None
        self.concentration_mapping = {}
        self.error_widget = error_widget

    def log_error(self, message):
        if self.error_widget:
            self.error_widget.append(message)
        else:
            print(message)

    def load_sheets(self):
        try:
            self.sheet1_df = pd.read_excel(self.file_path, sheet_name='Sheet1', header=None)
            self.sheet2_df = pd.read_excel(self.file_path, sheet_name='Sheet3', header=None)
            self.sheet4_df = pd.read_excel(self.file_path, sheet_name='Sheet4', header=None)
        except Exception as e:
            self.log_error(f"Error loading sheets: {e}")
            raise

    def create_concentration_mapping(self):
        for _, row in self.sheet2_df.iterrows():
            probes = row[:3].dropna().tolist()
            concentration = row[3]
            for probe in probes:
                self.concentration_mapping[probe] = concentration

    def map_concentrations_to_sheet1(self):
        concentration_row = []
        
        for col in self.sheet1_df.columns:
            found_concentration = None
            column_name = self.sheet1_df.iloc[0, col]

            for cell in self.sheet1_df[col]:
                if cell in self.concentration_mapping:
                    found_concentration = self.concentration_mapping[cell]
                    break

            if found_concentration is None:
                self.log_error(f"Concentration not found for column: {column_name}")

            concentration_row.append(found_concentration)

        concentration_df = pd.DataFrame([concentration_row], columns=self.sheet1_df.columns)
        return pd.concat([concentration_df, self.sheet1_df], ignore_index=True)

    def save_to_excel(self, df, file_path=None, sheet_name='Sheet1'):
        """Save DataFrame to Excel with proper mode handling"""
        file_path = file_path or self.file_path
        try:
            # Determine if we're appending to existing file or creating new
            if file_path == self.file_path:
                # Append mode with sheet replacement
                with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                # New file mode - no if_sheet_exists parameter needed
                with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            self.log_error(f"Saved to {file_path} (sheet: {sheet_name})")
        except Exception as e:
            self.log_error(f"Error saving to Excel: {e}")
            raise

    def process(self):
        try:
            # Original processing
            self.load_sheets()
            self.create_concentration_mapping()
            modified_sheet1 = self.map_concentrations_to_sheet1()
            self.save_to_excel(modified_sheet1)
            
            # NEW STEPS
           # 1. Replace column headers with values from row 3 (index 2)
            modified_sheet1.columns = modified_sheet1.iloc[1]

# 2. Delete original row 3 (now redundant since promoted as header)
            modified_sheet1 = modified_sheet1.drop(1).reset_index(drop=True)
            
            # 3. Save as 'kineticanalysis.xlsx'
            kinetic_file = "kineticanalysis.xlsx"
            self.save_to_excel(modified_sheet1, file_path=kinetic_file, sheet_name='Sheet1')
            
            # 4. Delete rows 2-4 (indices 1,2,3)
            modified_sheet1 = modified_sheet1.drop([1,2, 3]).reset_index(drop=True)
            
            # 5. Save final version to original file
            self.save_to_excel(modified_sheet1)

        except Exception as e:
            self.log_error(f"Error during processing: {e}")
            raise
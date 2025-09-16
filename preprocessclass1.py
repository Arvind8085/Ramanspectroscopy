import pandas as pd

class ExcelProcessor1:
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = pd.read_excel(input_file, sheet_name='Sheet2')
        self.selected_rows = pd.DataFrame(columns=self.df.columns)
    
    def find_closest_row(self, subset_df):
        target_value = 0.85
        closest_row_index = (subset_df['CCDLoadLevel:'] - target_value).abs().idxmin()
        return subset_df.loc[[closest_row_index]]  # Return as DataFrame
    
    def process_chunks(self):
        chunk = []
        in_chunk = False
        
        for index, row in self.df.iterrows():
            if "Start" in row['Name:']:
                in_chunk = True
                chunk = [row.to_frame().T]  # Store as DataFrame
            elif "Stop" in row['Name:']:
                if in_chunk:
                    chunk.append(row.to_frame().T)
                    subset_df = pd.concat(chunk, ignore_index=True)
                    closest_row = self.find_closest_row(subset_df)
                    
                    # Update Name: field
                    name_parts = [part for part in subset_df.iloc[0]['Name:'].split() if 'Start' not in part and 'Stop' not in part]
                    closest_row.loc[:, 'Name:'] = ' '.join(name_parts)
                    
                    # Append to selected_rows
                    self.selected_rows = pd.concat([self.selected_rows, closest_row], ignore_index=True)
                in_chunk = False
                chunk = []
            elif in_chunk:
                chunk.append(row.to_frame().T)
            else:
                self.selected_rows = pd.concat([self.selected_rows, row.to_frame().T], ignore_index=True)
    
    def write_to_excel(self):
        with pd.ExcelWriter(self.input_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            self.selected_rows.to_excel(writer, sheet_name='Sheet4', index=False)
    
    def run(self):
        self.process_chunks()
        self.write_to_excel()

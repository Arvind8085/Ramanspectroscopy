import pandas as pd
from itertools import combinations
from openpyxl import load_workbook

class ExcelDataProcessor:
    def __init__(self, input_file_path, grouped_output_path, ratios_output_path, analysis_output_path):
        self.input_file_path = input_file_path
        self.grouped_output_path = grouped_output_path
        self.ratios_output_path = ratios_output_path
        self.analysis_output_path = analysis_output_path
        self.grouped_dict = {}
        self.ratios_data = []

    def load_and_group_data(self):
        excel_file = pd.ExcelFile(self.input_file_path)
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(self.input_file_path, sheet_name=sheet_name)

            # Try to extract Probe and Punkt
            extracted = df['intensity_column'].str.extract(r'(Probe \d+)\s+(Punkt \d+)', expand=True)
            df['Probe'] = extracted[0]
            df['Punkt'] = extracted[1]

            # Handle rows where extraction failed
            for idx, row in df.iterrows():
                probe = row['Probe']
                punkt = row['Punkt']

                if pd.isna(probe) or pd.isna(punkt):
                    # Use intensity_column as fallback group key
                    fallback = str(row['intensity_column']).strip().replace(" ", "_")
                    probe = fallback
                    punkt = "unknown"
                    df.at[idx, 'Probe'] = probe
                    df.at[idx, 'Punkt'] = punkt

                group_key = f"{probe}_{punkt}"[:31]  # Excel sheet name limit
                if group_key not in self.grouped_dict:
                    self.grouped_dict[group_key] = []
                self.grouped_dict[group_key].append(row)



    def save_grouped_data(self):
        with pd.ExcelWriter(self.grouped_output_path) as writer:
            for group_key, group_rows in self.grouped_dict.items():
                group_df = pd.DataFrame(group_rows)
                sheet_name_out = group_key[:31]
                group_df.to_excel(writer, sheet_name=sheet_name_out, index=False)
        print(f"Grouped data saved to {self.grouped_output_path}")

    def calculate_ratios(self):
        for sheet_name in self.grouped_dict.keys():
            df = pd.DataFrame(self.grouped_dict[sheet_name])
            for (i, j) in combinations(df.index, 2):
                wavenumber_i = df.at[i, 'Wavenumber']
                wavenumber_j = df.at[j, 'Wavenumber']
                height_i = df.at[i, 'Peak Height']
                height_j = df.at[j, 'Peak Height']
                auc_i = df.at[i, 'AUC']
                auc_j = df.at[j, 'AUC']
                cluster_i = df.at[i, 'Clusters']
                cluster_j = df.at[j, 'Clusters']
                concentration_i = df.at[i, 'Concentration']
                if cluster_i == cluster_j:
                    continue
                if wavenumber_i < wavenumber_j:
                    wavenumber_i, wavenumber_j = wavenumber_j, wavenumber_i
                    height_i, height_j = height_j, height_i
                    auc_i, auc_j = auc_j, auc_i
                    cluster_i, cluster_j = cluster_j, cluster_i
                if wavenumber_j == 0 or height_j == 0 or auc_j == 0:
                    continue
                wavenumber_ratio = wavenumber_i / wavenumber_j
                peak_height_ratio = height_i / height_j
                auc_ratio = auc_i / auc_j
                self.ratios_data.append({
                    'Sheet': sheet_name,
                    'Wavenumber 1': wavenumber_i,
                    'Wavenumber 2': wavenumber_j,
                    'Wavenumber Ratio': wavenumber_ratio,
                    'Height 1': height_i,
                    'Height 2': height_j,
                    'Peak Height Ratio': peak_height_ratio,
                    'AUC 1': auc_i,
                    'AUC 2': auc_j,
                    'AUC Ratio': auc_ratio,
                    'Cluster 1': cluster_i,
                    'Cluster 2': cluster_j,
                    'Concentration': concentration_i
                })

    def save_ratios(self):
        ratios_df = pd.DataFrame(self.ratios_data)
        ratios_df.to_excel(self.ratios_output_path, sheet_name='Ratios', index=False)
        print(f"Ratios saved to {self.ratios_output_path}")

    def analyze_ratios(self):
        ratios_df = pd.DataFrame(self.ratios_data)
        analysis_data = []
        grouped_ratios = ratios_df.groupby(['Concentration', 'Cluster 1', 'Cluster 2'])
        for (concentration, cluster1, cluster2), group in grouped_ratios:
            mean_wavenumber_ratio = group['Wavenumber Ratio'].mean()
            std_wavenumber_ratio = group['Wavenumber Ratio'].std()
            mean_peak_height_ratio = group['Peak Height Ratio'].mean()
            std_peak_height_ratio = group['Peak Height Ratio'].std()
            mean_auc_ratio = group['AUC Ratio'].mean()
            std_auc_ratio = group['AUC Ratio'].std()
            wavenumber1 = round(group['Wavenumber 1'].iloc[0])
            wavenumber2 = round(group['Wavenumber 2'].iloc[0])
            analysis_data.append({
                'Concentration': concentration,
                'Cluster 1': cluster1,
                'Cluster 2': cluster2,
                'Mean Wavenumber Ratio': mean_wavenumber_ratio,
                'Std Wavenumber Ratio': std_wavenumber_ratio,
                'Mean Peak Height Ratio': mean_peak_height_ratio,
                'Std Peak Height Ratio': std_peak_height_ratio,
                'Mean AUC Ratio': mean_auc_ratio,
                'Std AUC Ratio': std_auc_ratio,
                'Wavenumber 1': wavenumber1,
                'Wavenumber 2': wavenumber2
            })
        analysis_df = pd.DataFrame(analysis_data)
        analysis_df.to_excel(self.analysis_output_path, sheet_name='Analysis', index=False)
        print(f"Analysis saved to {self.analysis_output_path}")

    def calculate_sum_of_mean_heights(self):
        ratios_df = pd.DataFrame(self.ratios_data)
        ratios_df['Concentration'] = ratios_df['Concentration'].apply(
            lambda x: float(str(x).replace("10^", "1E")))
        highest_concentration = ratios_df['Concentration'].max()
        filtered_df = ratios_df[ratios_df['Concentration'] == highest_concentration]
        
        if filtered_df.empty:
            print("No rows match the highest concentration filter.")
            return

        if 'Height 1' not in filtered_df.columns or 'Height 2' not in filtered_df.columns:
            print("Height columns are missing in the data.")
            return

        mean_peak_heights = filtered_df.groupby(['Cluster 1', 'Cluster 2']).agg(
            Mean_Height_1=('Height 1', 'mean'),
            Mean_Height_2=('Height 2', 'mean')
        ).reset_index()
        mean_peak_heights['Sum of Mean Heights'] = (
            mean_peak_heights['Mean_Height_1'] + mean_peak_heights['Mean_Height_2'])
        
        # Corrected approach using ExcelWriter in append mode
        with pd.ExcelWriter(self.analysis_output_path, 
                        engine='openpyxl', 
                        mode='a', 
                        if_sheet_exists='replace') as writer:
            mean_peak_heights.to_excel(writer, sheet_name='Sum of Mean Heights', index=False)
        
        print(f"Sum of mean peak heights saved to {self.analysis_output_path}")

    def update_analysis_with_sum(self):
        # Read existing data
        analysis_df = pd.read_excel(self.analysis_output_path, sheet_name='Analysis')
        sum_heights_df = pd.read_excel(self.analysis_output_path, sheet_name='Sum of Mean Heights')
        
        # Process concentration values
        analysis_df['Concentration'] = analysis_df['Concentration'].apply(
            lambda x: float(str(x).replace("10^", "1E")))
        
        highest_concentration = analysis_df['Concentration'].max()
        analysis_df_high_conc = analysis_df[analysis_df['Concentration'] == highest_concentration]
        
        # Merge with sum data
        analysis_df_high_conc = pd.merge(
            analysis_df_high_conc, 
            sum_heights_df[['Cluster 1', 'Cluster 2', 'Sum of Mean Heights']],
            on=['Cluster 1', 'Cluster 2'], 
            how='left'
        )
        
        # Combine with other concentrations
        updated_analysis = pd.concat([
            analysis_df[analysis_df['Concentration'] != highest_concentration], 
            analysis_df_high_conc
        ])
        
        # Save with corrected approach
        with pd.ExcelWriter(self.analysis_output_path, engine='openpyxl') as writer:
            updated_analysis.to_excel(writer, sheet_name='Analysis', index=False)
            if 'Sum of Mean Heights' in writer.book.sheetnames:
                sum_heights_df.to_excel(writer, sheet_name='Sum of Mean Heights', index=False)
        
        print(f"Updated analysis saved to {self.analysis_output_path}")



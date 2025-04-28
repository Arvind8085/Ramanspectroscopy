from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QPushButton,QScrollArea,
    QHBoxLayout, QSpinBox, QLabel, QTextEdit, QLineEdit, QGridLayout, QFileDialog, QComboBox, QListWidget, QAbstractItemView
)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from Baseline_and_filters import apply_butterworth, baseline_als_optimized
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.integrate import simps
from sklearn.mixture import GaussianMixture
from PyQt5.QtWidgets import (

    QLabel, QFrame  # Add QLabel for error display, QFrame for layout division
)
from spzexceltab0 import Preprocess
from PyQt5.QtWidgets import QSplitter
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import pyqtSignal
from sorting_AUC_25 import ExcelDataProcessor
from sumofheightsclass import ClusterAnalysisPlotter
from GUI_Tab_3 import Tab_3
from AUCclass import NitroPhenolAnalysis
from tab4_peak_vs_concentration import Tab_4
from meanclass import ClusterStatisticsProcessor
import os

def asymmetric_pseudo_voigt(x, amplitude, center, sigma_g, sigma_l, asymmetry, eta):
    try:
        sigma_g_left = sigma_g * (1 + asymmetry)
        sigma_g_right = sigma_g * (1 - asymmetry)
        sigma_l_left = sigma_l * (1 + asymmetry)
        sigma_l_right = sigma_l * (1 - asymmetry)

        left_side = (x < center)
        g_left = np.exp(-((x[left_side] - center) ** 2) / (2 * sigma_g_left ** 2))
        l_left = (sigma_l_left ** 2) / ((x[left_side] - center) ** 2 + sigma_l_left ** 2)
        g_right = np.exp(-((x[~left_side] - center) ** 2) / (2 * sigma_g_right ** 2))
        l_right = (sigma_l_right ** 2) / ((x[~left_side] - center) ** 2 + sigma_l_right ** 2)

        pseudo_voigt = np.zeros_like(x)
        pseudo_voigt[left_side] = (1 - eta) * g_left + eta * l_left
        pseudo_voigt[~left_side] = (1 - eta) * g_right + eta * l_right

        return amplitude * pseudo_voigt

    except Exception as e:
        raise RuntimeError(f"Error in pseudo-Voigt calculation: {str(e)}")


class PeakAnalysisApp(QMainWindow):
    file_saved_signal = pyqtSignal(str)  # Signal to notify file save
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PeakAnalysis")
        self.setGeometry(100, 100, 1200, 800)

        self.graphs = []  # List of graphs to navigate through
        self.data_file_path = None  # Initialize with None
        self.current_graph_index = 0  # Current graph index
        self.file_path = ""
        self.df_results = None
        self.detected_wavenumbers = []
        self.peak_heights = []
        self.peak_concentrations = []
        self.peak_aucs = []
        self.cluster_labels = []
        self.intensity_columns1 = []  # Initialize peak areas for later use
        self.file_paths = []  # Store all loaded file paths
        self.data_file_path = None  # Currently selected file path
        self.data_file_paths = []  # Initialize as an empty list
        self.processed_files = {}
        # Main container
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab 1: Graph Navigation
        self.tab0= Preprocess()
        self.tab1 = QWidget()
        self.tab4= Tab_4()
        self.tab2 = QWidget()
        self.gui2 = Tab_3()
        

    
        self.tab0.init_tab0(self.tabs)
        self.tabs.addTab(self.tab1, "Spectral Analysis")
        self.tab4.init_tab4(self.tabs)
        self.tabs.addTab(self.tab2, "Ratios Plotting")

        
        self.init_tab1()
        self.init_tab2()
        self.gui2.init_tab3(self.tabs)  # Call inherited `init_tab3`
        
    def init_tab1(self):
        main_layout = QVBoxLayout()
        
        # Create a main splitter for dividing the graph section and parameters/error section
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Create a splitter for graphs within the left section of the main splitter
        graph_splitter = QSplitter(Qt.Vertical)
        graph_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Left graph setup
        self.figure_left = plt.figure()
        self.canvas_left = FigureCanvas(self.figure_left)
        toolbar_left = NavigationToolbar(self.canvas_left, self)

        left_layout = QVBoxLayout()
        left_layout.addWidget(toolbar_left)
        left_layout.addWidget(self.canvas_left)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # Right graph setup
        self.figure_right = plt.figure()
        self.canvas_right = FigureCanvas(self.figure_right)
        toolbar_right = NavigationToolbar(self.canvas_right, self)

        right_layout = QVBoxLayout()
        right_layout.addWidget(toolbar_right)
        right_layout.addWidget(self.canvas_right)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        # Add the left and right graphs to the graph splitter
        graph_splitter.addWidget(left_widget)
        graph_splitter.addWidget(right_widget)
        
        # Add the graph splitter to the main splitter
        main_splitter.addWidget(graph_splitter)
        
        # Bottom-right parameters and error layout
        bottom_layout = QVBoxLayout()
        
        # Parameters layout
        parameter_layout = QGridLayout()
        self.smoothness_input = QLineEdit()
        self.smoothness_input.setPlaceholderText("Baseline Smoothness")
        parameter_layout.addWidget(QLabel("Baseline Smoothness"), 0, 0)
        parameter_layout.addWidget(self.smoothness_input, 0, 1)

        self.p_input = QLineEdit()
        self.p_input.setPlaceholderText("Baseline p")
        parameter_layout.addWidget(QLabel("Baseline Parameter (p)"), 1, 0)
        parameter_layout.addWidget(self.p_input, 1, 1)

        self.butter_order_input = QLineEdit()
        self.butter_order_input.setPlaceholderText("Butterworth Order")
        parameter_layout.addWidget(QLabel("Butterworth Order:"), 2, 0)
        parameter_layout.addWidget(self.butter_order_input, 2, 1)

        self.cutoff_freq_input = QLineEdit()
        self.cutoff_freq_input.setPlaceholderText("Cutoff Frequency")
        parameter_layout.addWidget(QLabel("Cutoff Frequency"), 3, 0)
        parameter_layout.addWidget(self.cutoff_freq_input, 3, 1)

        self.gmm_type_input = QComboBox()
        self.gmm_type_input.addItems(["full", "diag", "tied", "spherical"])
        parameter_layout.addWidget(QLabel("GMM Type:"), 4, 0)
        parameter_layout.addWidget(self.gmm_type_input, 4, 1)

        self.random_seed_input = QSpinBox()
        self.random_seed_input.setRange(0, 1000)
        parameter_layout.addWidget(QLabel("Random Seed:"), 5, 0)
        parameter_layout.addWidget(self.random_seed_input, 5, 1)

        nav_buttons_layout = QVBoxLayout()
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load_data)
        nav_buttons_layout.addWidget(self.load_button)
        
        self.process_button = QPushButton("Run")
        self.process_button.clicked.connect(self.process_data)
        nav_buttons_layout.addWidget(self.process_button)
        self.process_all_button = QPushButton("Process All")
        #self.process_all_button.clicked.connect(self.process_all_files)
        nav_buttons_layout.addWidget(self.process_all_button)
        
        self.show_combined_button = QPushButton("Show Combined Graphs")
        #self.show_combined_button.clicked.connect(self.show_combined_graphs)
        nav_buttons_layout.addWidget(self.show_combined_button)
        self.savespec_button = QPushButton("Save Spectral Data")
        self.savespec_button.clicked.connect(self.save_results)
        nav_buttons_layout.addWidget(self.savespec_button)
        
        self.run_button = QPushButton("Save Ratios Data")
        self.run_button.clicked.connect(self.run_processor_methods)
        nav_buttons_layout.addWidget(self.run_button)
        
        self.error_widget = QTextEdit()
        self.error_widget.setReadOnly(True)
        self.error_widget.setStyleSheet("color: red;")
        self.error_widget.setPlaceholderText("Errors and messages will appear here.")
        # Add Dataset Selector
        self.dataset_selector = QListWidget()  # Multi-item list view
        self.dataset_selector.setSelectionMode(QAbstractItemView.MultiSelection)
        self.dataset_selector.itemSelectionChanged.connect(self.file_selection_changed)
        
        
        

        # Combine parameter layout and navigation buttons
        param_nav_layout = QHBoxLayout()
        param_nav_layout.addLayout(parameter_layout)
        param_nav_layout.addLayout(nav_buttons_layout)
        param_nav_layout.addWidget(QLabel("Select Dataset:"))
        param_nav_layout.addWidget(self.dataset_selector)
        
        
        
        bottom_layout.addLayout(param_nav_layout)
        bottom_layout.addWidget(self.error_widget)
        
        # Add the bottom layout to the main splitter
        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom_layout)
        main_splitter.addWidget(bottom_widget)
        
        # Adjust the size ratio between the graphs and parameters/error section
        main_splitter.setStretchFactor(0, 7)  # 70% for graphs
        main_splitter.setStretchFactor(1, 3)  # 30% for parameters/error
        
        # Create a layout for the "Next" and "Previous" buttons
        nav_buttons_layout_below = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_graph)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_graph)
        
        nav_buttons_layout_below.addWidget(self.prev_button)
        nav_buttons_layout_below.addWidget(self.next_button)
        
        # Add the main splitter and navigation buttons layout to the main layout
        main_layout.addWidget(main_splitter)
        main_layout.addLayout(nav_buttons_layout_below)
        
        # Set the main layout for the tab
        self.tab1.setLayout(main_layout)



    def init_error_tab(self):
        main_layout = QVBoxLayout()
        self.error_widget = QLabel("")
        self.error_widget.setStyleSheet("color: red; font-weight: bold;")
        self.error_widget.setWordWrap(True)
        self.error_widget.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        main_layout.addWidget(self.error_widget)
        self.error_tab.setLayout(main_layout)

    def init_tab2(self):
        main_layout = QVBoxLayout()

        # Initialize matplotlib figure and canvas for plot
        self.figure_tab2 = plt.figure()
        self.canvas_tab2 = FigureCanvas(self.figure_tab2)
        self.canvas_tab2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Initialize matplotlib figure and canvas for plot5
        self.figure_tab5 = plt.figure()
        self.canvas_tab5 = FigureCanvas(self.figure_tab5)
        self.canvas_tab5.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Graph navigation toolbars
        self.toolbar_tab2 = NavigationToolbar(self.canvas_tab2, self)
        self.toolbar_tab5 = NavigationToolbar(self.canvas_tab5, self)

        # Navigation buttons for Plot 2
        graph_nav_layout_tab2 = QHBoxLayout()
        self.prev_button_tab2 = QPushButton("Previous")
        self.prev_button_tab2.clicked.connect(self.show_previous_graph_tab2)
        self.next_button_tab2 = QPushButton("Next")
        self.next_button_tab2.clicked.connect(self.show_next_graph_tab2)
        graph_nav_layout_tab2.addWidget(self.prev_button_tab2)
        graph_nav_layout_tab2.addWidget(self.next_button_tab2)

        # Navigation buttons for Plot 5
        graph_nav_layout_tab5 = QHBoxLayout()
        self.prev_button_tab5 = QPushButton("Previous")
        self.prev_button_tab5.clicked.connect(self.show_previous_graph_tab5)
        self.next_button_tab5 = QPushButton("Next")
        self.next_button_tab5.clicked.connect(self.show_next_graph_tab5)
        graph_nav_layout_tab5.addWidget(self.prev_button_tab5)
        graph_nav_layout_tab5.addWidget(self.next_button_tab5)

        # Unified Plot button
        self.plot_button_both = QPushButton("Plot Both")
        self.plot_button_both.clicked.connect(self.plot_both)

        # Error widgets for Tab 2
        self.error_widget_tab2 = QTextEdit()
        self.error_widget_tab2.setReadOnly(True)
        self.error_widget_tab2.setStyleSheet("color: red;")
        self.error_widget_tab2.setPlaceholderText("Errors and messages for Peak heights will appear here.")
        self.error_widget_tab2.setFixedHeight(50)  # Reduced height

        self.error_widget_tab5 = QTextEdit()
        self.error_widget_tab5.setReadOnly(True)
        self.error_widget_tab5.setStyleSheet("color: red;")
        self.error_widget_tab5.setPlaceholderText("Errors and messages for AUC will appear here.")
        self.error_widget_tab5.setFixedHeight(50)  # Reduced height

        # Layout for Plot 2
        content_layout_tab2 = QVBoxLayout()
        content_layout_tab2.addWidget(self.toolbar_tab2)
        content_layout_tab2.addWidget(self.canvas_tab2, stretch=3)
        content_layout_tab2.addWidget(self.error_widget_tab2)
        content_layout_tab2.addLayout(graph_nav_layout_tab2)

        # Layout for Plot 5
        content_layout_tab5 = QVBoxLayout()
        content_layout_tab5.addWidget(self.toolbar_tab5)
        content_layout_tab5.addWidget(self.canvas_tab5, stretch=3)
        content_layout_tab5.addWidget(self.error_widget_tab5)
        content_layout_tab5.addLayout(graph_nav_layout_tab5)

        # Wrap each layout in a QWidget for QSplitter
        left_widget = QWidget()
        left_widget.setLayout(content_layout_tab2)

        right_widget = QWidget()
        right_widget.setLayout(content_layout_tab5)

        # Create a QSplitter to hold the two widgets
        splitter = QSplitter(Qt.Horizontal)  # Horizontal splitter for left and right graphs
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        # Add the splitter and the unified Plot Both button to the main layout
        main_layout.addWidget(splitter)
        main_layout.addWidget(self.plot_button_both)  # Plot Both button at the bottom

        # Set the main layout for Tab 2
        self.tab2.setLayout(main_layout)


    def safe_execute(self, func):
        def wrapper():
            try:
                func()
            except Exception as e:
                 self.error_widget.setText(f"Error: {str(e)}")
        return wrapper
    def load_data(self):
        # Open file dialog to select multiple Excel files
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open Excel Files", "", "Excel Files (*.xlsx)")
        if file_paths:
            # Append new files to the existing list instead of overwriting
            self.file_paths.extend(file_paths)
            self.file_paths = list(set(self.file_paths))  # Remove duplicates if needed

            # Inform the user of successful loading
            self.error_widget.setText("Files loaded successfully!")

            # Update the QListWidget with the loaded file paths
            self.dataset_selector.clear()
            self.dataset_selector.addItems(self.file_paths)
        else:
            self.error_widget.setText("No files uploaded.")

    def file_selection_changed(self):
        # Get the selected items from the QListWidget
        selected_items = self.dataset_selector.selectedItems()
        if selected_items:
            # Update `self.data_file_path` to the first selected file
            self.data_file_path = selected_items[0].text()
            
            # Pass the file path to Tab 4 if `set_file_path` exists
            if hasattr(self.tab4, 'set_file_path') and callable(self.tab4.set_file_path):
                self.tab4.set_file_path(self.data_file_path)
            
            # Optional: Inform the user about the selected file
            self.error_widget.setText(f"Selected file: {self.data_file_path}")
        else:
            # Clear the current file path if nothing is selected
            self.data_file_path = None
            self.error_widget.setText("No file selected.")


    
    def process_data(self, filepath =None, processed_output_path = None):
        file_path = filepath or self.data_file_path
        if not file_path:
            self.error_widget.setText("No file selected. Please select a file before processing.")
            return
        
        # Process the selected file
        #self.error_widget.setText(f"Processing file: {self.data_file_path}")
        # Add your processing logic here
        
        try:
            baseline_smoothness = float(self.smoothness_input.text() or 1e8)  # Default smoothness
            baseline_p = float(self.p_input.text() or 0.00001)  # Default p
            butterworth_order = int(self.butter_order_input.text() or 9)
            cutoff_frequency = float(self.cutoff_freq_input.text() or 0.1)
            gmm_type = self.gmm_type_input.currentText()
            random_seed = self.random_seed_input.value()
             # Retrieve and validate parameters from Tab 1 inputs
            self.baseline_smoothness = float(self.smoothness_input.text() or 1e8)
            self.baseline_p = float(self.p_input.text() or 0.00001)
            self.butterworth_order = int(self.butter_order_input.text() or 9)
            self.cutoff_frequency = float(self.cutoff_freq_input.text() or 0.1)
            self.gmm_type = self.gmm_type_input.currentText()
            self.random_seed = self.random_seed_input.value()
        except ValueError:
            self.error_widget.setText("Invalid parameter input. Please enter valid numbers.")
            return


        df = pd.read_excel(file_path)
        

        # Extract the row with concentrations (first row) and drop it from the main DataFrame
        concentration_row = df.iloc[0]
        df = df.drop(0).reset_index(drop=True)

        # Select intensity columns and map each to its concentration from the extracted row
        intensity_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['spectrum', 'faser', 'ntp', 'probe','Start','Stop','1','2','3','4','6','7'])]
        concentration_mapping = {col: concentration_row[col] for col in intensity_columns}

        self.graphs = []
        # Clear previously stored results
        self.graph_data = []  # Store data for each graph
        self.detected_wavenumbers = []  # Reset results for new processing
        self.peak_heights = []
        self.peak_concentrations = []
        self.peak_aucs = []
        self.cluster_labels = []
        self.intensity_columns1 = []  # Clear any existing graphs

        for intensity_column in intensity_columns:
            intensity_data = df[intensity_column].tolist()
            wavenumber_data = df['Wavenumber'].tolist()

            concentration = concentration_mapping[intensity_column]

            # Data preparation and filtering
            data = np.array([intensity_data, wavenumber_data]).T
            filtered_data = data[(data[:, 1] >= 355) & (data[:, 1] <= 1900)]
            original_intensity = filtered_data[:, 0]

            # Apply Butterworth filter and baseline correction
            order = butterworth_order
            cutoff1_frequency = cutoff_frequency
            filtered_intensity = apply_butterworth(original_intensity, order, cutoff1_frequency)
            baseline = baseline_als_optimized(filtered_intensity, smth=baseline_smoothness, p=baseline_p)
            baseline_corrected_intensity = original_intensity - baseline
            filtered_intensity_2 = apply_butterworth(baseline_corrected_intensity, order, cutoff1_frequency)
            baseline2 = baseline_als_optimized(filtered_intensity_2, smth=baseline_smoothness, p=baseline_p)
            noise_level2 = np.std(filtered_intensity - original_intensity) + baseline2
            above_noise_peaks, _ = find_peaks(filtered_intensity_2, height=noise_level2)
            above_noise_x = filtered_data[:, 1][above_noise_peaks]
            above_noise_data1 = filtered_intensity_2[above_noise_peaks]

            fitted_curves = []
            for peak_index in range(len(above_noise_peaks)):
                x_peak = above_noise_x[peak_index]
                y_peak = above_noise_data1[peak_index]
                fit_range = (filtered_data[:, 1] >= (x_peak - 20)) & (filtered_data[:, 1] <= (x_peak + 20))
                x_data_fit = filtered_data[:, 1][fit_range]
                y_data_fit = filtered_intensity_2[fit_range]

                initial_guess = [y_peak, x_peak, 5, 5, 0, 0.5]
                bounds = ([0, x_peak - 20, 0, 0, -1, 0], [y_peak * 2, x_peak + 20, 50, 50, 1, 1])
                popt, pcov = curve_fit(asymmetric_pseudo_voigt, x_data_fit, y_data_fit, p0=initial_guess, bounds=bounds)
                fitted_curve = asymmetric_pseudo_voigt(filtered_data[:, 1], *popt)
                fitted_curves.append(fitted_curve)
                # Calculate the area under the fitted curve
                auc = simps(fitted_curve, filtered_data[:, 1])

                self.detected_wavenumbers.append(x_peak)
                self.peak_heights.append(y_peak)
                self.peak_concentrations.append(concentration)
                self.peak_aucs.append(auc)
                self.intensity_columns1.append(intensity_column)
           

            self.graphs.append((
                filtered_data[:, 1],  # wavenumber
                original_intensity,  # original intensity
                baseline_corrected_intensity,  # baseline corrected intensity
                filtered_intensity,  # filtered intensity
                baseline,
                filtered_intensity_2,  # corrected filtered intensity
                baseline2,  # second baseline correction
                noise_level2,  # noise level
                above_noise_x,  # peak x values
                above_noise_data1,  # peak y values
                fitted_curves,
                intensity_column,
                concentration
            ))
            
    
        # GMM Clustering
        
        peak_data = np.array([self.detected_wavenumbers, self.peak_heights]).T
        gmm_type = self.gmm_type_input.currentText()
        random_seed = self.random_seed_input.value()
        lowest_bic = np.inf
        best_gmm = None
        bic_values = []
        for n_clusters in range(1, 60):
            gmm = GaussianMixture(n_components=n_clusters, covariance_type=gmm_type, init_params='kmeans', random_state=random_seed)
            gmm.fit(peak_data)
            bic = gmm.bic(peak_data)
            bic_values.append(bic)
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm
                
        optimal_clusters = best_gmm.n_components
        cluster_labels = best_gmm.predict(peak_data)
        self.cluster_labels = cluster_labels

        # Display the optimal number of clusters in the error widget
        self.error_widget.append(f"Optimal number of clusters selected: {optimal_clusters}. Peak analysis completed, you may save the data for further processing.")
        
        
        self.current_graph_index = 0
        self.gmm_results = {
            "peak_data": peak_data,
            "cluster_labels": cluster_labels
        }

        # Save results in self.processed_files
        self.processed_files[file_path] = {
            "graphs": self.graphs.copy(),
            "gmm_results": self.gmm_results.copy(),
            # Add other data you want to keep for each file
        }

            # Save results with cluster statistics
        df_results = pd.DataFrame({
            'Intensity_Column': self.intensity_columns1,
            'Wavenumber': self.detected_wavenumbers,
            'Peak Height': self.peak_heights,
            'Concentration': self.peak_concentrations,
            'AUC': self.peak_aucs,
            'Cluster': cluster_labels
        })

        grouped_1 = df_results.groupby(['Cluster', 'Concentration']).agg(
            mean_wavenumber=('Wavenumber', 'mean'),
            mean_height=('Peak Height', 'mean'),
            mean_AUC=('AUC', 'mean'),
            std_wavenumber=('Wavenumber', 'std'),
            std_height=('Peak Height', 'std'),
            std_AUC=('AUC', 'std')
        ).reset_index()

        output_file1 = 'peak_data_before_statistics.xlsx'
        df_results.to_excel(output_file1, sheet_name='Dataset 1 Peaks', index=False)
        self.error_widget.append(f'Peak data saved to {output_file1}')

        output_file2 = 'peak_statistics_R4.xlsx'
        grouped_1.to_excel(output_file2, sheet_name='Dataset 1 Stats', index=False)
        self.error_widget.append(f'Statistics saved to {output_file2}')

        cluster_processor = ClusterStatisticsProcessor(output_file2, sheet_name='Dataset 1 Stats')
        cluster_processor.load_data()

            # Process the column(s) and generate the new output
        cluster_processor.process_column('mean_wavenumber')
        processed_output_path = 'processed_cluster_statistics.xlsx'
        cluster_processor.save_to_file(processed_output_path)
        self.processed_statistics_output_path = processed_output_path
        self.tab4.PlotBoth2(processed_output_path)
        self.error_widget.append(f'Statistics saved to {processed_output_path}')
        
        
        self.show_graph()

    

    def show_graph(self):
        if not self.graphs:
            return

         # Clear both figures
        self.figure_left.clear()
        self.figure_right.clear()
          # Left plot: Spectral analysis
      
        ax1 = self.figure_left.add_subplot(111)

        # Get current graph data
        (
            wavenumber,
            original_intensity,
            baseline_corrected_intensity,
            filtered_intensity,
            baseline,
            
            filtered_intensity_2,
            baseline2,
            noise_level2,
            above_noise_x,
            above_noise_data1,
            fitted_curves,
            intensity_column,
            concentration
        ) = self.graphs[self.current_graph_index]

        ax1.plot(wavenumber, original_intensity, label="Original Intensity")
        ax1.plot(wavenumber, baseline_corrected_intensity, label="Baseline Corrected Intensity")
        ax1.plot(wavenumber, filtered_intensity, label="Filtered Intensity")
        ax1.plot(wavenumber, baseline, label="Baseline Corrected Intensity 1")
        ax1.plot(wavenumber, filtered_intensity_2, label="Filtered Intensity Corrected")
        ax1.plot(wavenumber, baseline2, label="Baseline Corrected Intensity 2")
        ax1.plot(wavenumber, noise_level2, label="Noise Level", linestyle="--", color="blue")
        ax1.plot(above_noise_x, above_noise_data1, "ro", label="Peaks")

        for fitted_curve in fitted_curves:
            ax1.plot(wavenumber, fitted_curve, linestyle="--")
            ax1.fill_between(wavenumber, fitted_curve, alpha=0.3)

        ax1.set_title(f"Sample: {intensity_column}, Concentration: {concentration} M")
        ax1.set_xlabel("Wavenumber")
        ax1.set_ylabel("Intensity")
        ax1.legend(
        loc='upper left',          # Position the legend
        #bbox_to_anchor=(1, 1),     # Place it outside the plot
        fontsize='8',          # Reduce font size
        ncol=4                     # Split into two columns
        )

        ax2 = self.figure_right.add_subplot(111)
        cluster_labels = self.gmm_results["cluster_labels"]
        
        for label in np.unique(cluster_labels):
            cluster_mask = cluster_labels == label
            peak_data = self.gmm_results["peak_data"]
            scatter = ax2.scatter(peak_data[cluster_mask, 0], peak_data[cluster_mask, 1], label=f'Cluster {label}', alpha=0.7)
        ax2.set_title("GMM Clustering")
        ax2.set_xlabel("Wavenumber")
        ax2.set_ylabel("Peak Height")
        ax2.legend(
        loc='upper left',          # Position the legend
        #bbox_to_anchor=(1, 1),     # Place it outside the plot
        fontsize='5',          # Reduce font size
        ncol=4                     # Split into two columns
        )

         
        
        self.canvas_left.draw()
        self.canvas_right.draw()

   


    def show_next_graph(self):
        if self.current_graph_index < len(self.graphs) - 1:
            self.current_graph_index += 1
            self.show_graph()

    def show_previous_graph(self):
        if self.current_graph_index > 0:
            self.current_graph_index -= 1
            self.show_graph()


    def save_results(self):
    # Check if results are available
        if not self.detected_wavenumbers or not isinstance(self.detected_wavenumbers, list):
            self.error_widget.setText("No results to save. Run the analysis first.")
            return

        # Ensure all result arrays are of the same length
        lengths = [
            len(self.detected_wavenumbers),
            len(self.peak_heights),
            len(self.cluster_labels),
            len(self.peak_concentrations),
            len(self.peak_aucs),
        ]
        if len(set(lengths)) > 1:
            self.error_widget.setText("Mismatch in results array lengths. Check data processing.")
            return

        # Open save file dialog
        output_file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "Excel Files (*.xlsx)"
        )
        if not output_file_path:
            self.error_widget.setText("Save operation canceled.")
            return

        # Prepare data for saving
        results_df = pd.DataFrame({
            "intensity_column": self.intensity_columns1,
            "Wavenumber": self.detected_wavenumbers,
            "Peak Height": self.peak_heights,
            "Concentration": self.peak_concentrations,
            "AUC": self.peak_aucs,
            "Clusters": self.cluster_labels

        })

        # Save the DataFrame to an Excel file
        try:
            results_df.to_excel(output_file_path, index=False)
            self.saved_file_path = output_file_path  # Store the path of the saved file
            self.error_widget.setText(f"Results successfully saved to {output_file_path}.")
        except Exception as e:
            self.error_widget.setText(f"Error saving results: {str(e)}")

    def run_processor_methods(self):
        print("Running processor methods...")
        if not hasattr(self, 'saved_file_path') or not self.saved_file_path:
            self.error_widget.setText("No saved file found. Save results first.")
            return

        try:
        # Ask the user where to save the final Ratios analysis file
            analysis_output_path, _ = QFileDialog.getSaveFileName(
                self, "Save Ratios Analysis Data", "Ratios_analysis.xlsx", "Excel Files (*.xlsx)"
            )
            if not analysis_output_path:
                self.error_widget.setText("Operation canceled: No path for Ratios Analysis.")
                return
            # Ensure correct file paths are specified
            processor = ExcelDataProcessor(
                input_file_path=self.saved_file_path,
                grouped_output_path='Grouped_data.xlsx',
                ratios_output_path='Data_Ratios_data_.xlsx',
                analysis_output_path=analysis_output_path
            )

        # Execute all processing methods
            processor.load_and_group_data()
            processor.save_grouped_data()
            processor.calculate_ratios()
            processor.save_ratios()
            processor.analyze_ratios()
            processor.calculate_sum_of_mean_heights()
            processor.update_analysis_with_sum()
            # Save the analysis output path for the plot method
            self.analysis_output_path = analysis_output_path
            self.gui2.set_analysis_output_path(self.analysis_output_path)
            self.error_widget.setText("All methods executed successfully.")
            print("All methods executed successfully.")

            
        except Exception as e:
            self.error_widget.setText(f"An error occurred: {e}")
            print(f"An error occurred: {e}")


    def plot_both(self):
        print("Plotting both graphs...")
        self.error_widget_tab2.clear()
        self.error_widget_tab5.clear()

        self.Plot()
        self.plot5()
            
    def Plot(self):
        print("Ploting executed.")
        
        if not hasattr(self, 'analysis_output_path') or not self.analysis_output_path:
            self.error_widget_tab2.setText("No analysis file found. Run the analysis first.")
            print("No analysis file found for Mean peak.")
            return

        try:
            # Clear the error widget before starting
            self.error_widget_tab2.clear()
            print("Cleared error widget.")

            plotter = ClusterAnalysisPlotter(self.analysis_output_path)
            plotter.load_and_prepare_data()
            print("Loaded and prepared data.")

            # Generate the plots and store both figures and subset details
            self.current_figures = []  # Store figures
            self.current_subsets = []  # Store subset details
            for fig, subset_details in plotter.generate_plots(top_n=60):
                if fig:
                    self.current_figures.append(fig)
                    self.current_subsets.append(subset_details)
                    print(f"Generated plot with R²={subset_details.split('R²=')[-1]}")
                else:
                    print(f"Failed to generate plot: {subset_details}")

            print(f"Total figures generated: {len(self.current_figures)}")

            if self.current_figures:
                self.current_figure_index = 0
                self.show_current_graph_tab2()
                self.error_widget_tab2.append("Plots successfully generated.")
                print("Plots successfully generated.")
            else:
                self.error_widget_tab2.append("No plots were generated.")
                print("No plots were generated.")
        except Exception as e:
            self.error_widget_tab2.setText(f"An error occurred while plotting: {e}")
            print(f"An error occurred while plotting: {e}")



    def show_current_graph_tab2(self):
        if hasattr(self, 'current_figures') and self.current_figures:
            print(f"Displaying figure {self.current_figure_index + 1}/{len(self.current_figures)}")
                
                # Assign the current figure to the canvas
            self.canvas_tab2.figure = self.current_figures[self.current_figure_index]
            self.canvas_tab2.draw_idle()

                # Update the error widget with subset details
            subset_details = self.current_subsets[self.current_figure_index]
            self.error_widget_tab2.setText(subset_details)
            print(f"Displayed figure {self.current_figure_index + 1}: {subset_details}")
        else:
            self.error_widget_tab2.setText("No figures to display.")
            print("No figures to display.")

    def show_previous_graph_tab2(self):
        if hasattr(self, 'current_figures') and self.current_figures:
            self.current_figure_index = (self.current_figure_index - 1) % len(self.current_figures)
            self.show_current_graph_tab2()

    def show_next_graph_tab2(self):
        if hasattr(self, 'current_figures') and self.current_figures:
            self.current_figure_index = (self.current_figure_index + 1) % len(self.current_figures)
            self.show_current_graph_tab2()

    def plot5(self):
        print("Ploting executed for AUC.")
        
        if not hasattr(self, 'analysis_output_path') or not self.analysis_output_path:
            self.error_widget_tab2.setText("No analysis file found. Run the analysis first.")
            print("No analysis file found for AUC.")
            return

        try:                                                                              
            # Clear the error widget before starting
            self.error_widget_tab5.clear()
            self.error_widget_tab5.append("Starting AUC analysis...")
            print("Cleared error widget.")

            plot5_analysis = NitroPhenolAnalysis(self.analysis_output_path, output_dir2=None)
            plot5_analysis.load_data()
            print("Loaded and prepared data for AUC analysis.")

            self.current_figures_plot5 = []
            self.current_subsets_plot5 = []

            for fig, subset_details in plot5_analysis.plot_all_ratios(top_n=60):
                if fig:
                    self.current_figures_plot5.append(fig)
                    self.current_subsets_plot5.append(subset_details)
                    print(f"Generated plot with R²={subset_details.split('R²=')[-1]}")
                else:
                    print(f"Failed to generate plot: {subset_details}")

        
            if self.current_figures_plot5:
                self.current_figure_index_plot5 = 0
                self.show_current_graph_plot5()
                self.error_widget_tab5.append("Plots successfully generated.")
                print("Plots successfully generated.")
            else:
                self.error_widget_tab5.append("No plots were generated.")
                print("No plots were generated.")
        except Exception as e:
            self.error_widget_tab5.setText(f"An error occurred while plotting: {e}")
            print(f"An error occurred while plotting: {e}")
    
       
    def show_current_graph_plot5(self):

        if hasattr(self, 'current_figures_plot5') and self.current_figures_plot5:
            print(f"Displaying figure Plot5 {self.current_figure_index_plot5 + 1}/{len(self.current_figures_plot5)}")

            self.canvas_tab5.figure = self.current_figures_plot5[self.current_figure_index_plot5]
            self.canvas_tab5.draw_idle()

            subset_details = (
                self.current_subsets_plot5[self.current_figure_index_plot5]
                if self.current_figure_index_plot5 < len(self.current_subsets_plot5)
                else "Subset details not available."
            )
            self.error_widget_tab5.setText(subset_details)
            print(f"Displayed Plot5 figure {self.current_figure_index_plot5 + 1}: {subset_details}")
        else:
            self.error_widget_tab5.setText("No figures to display for AUCS.")
            print("No figures to display AUCS.")

    def show_previous_graph_tab5(self):
        if hasattr(self, 'current_figures_plot5') and self.current_figures_plot5:
            self.current_figure_index_plot5 = (self.current_figure_index_plot5 - 1) % len(self.current_figures_plot5)
            self.show_current_graph_plot5()

    def show_next_graph_tab5(self):
        if hasattr(self, 'current_figures_plot5') and self.current_figures_plot5:
            self.current_figure_index_plot5 = (self.current_figure_index_plot5 + 1) % len(self.current_figures_plot5)
            self.show_current_graph_plot5()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = PeakAnalysisApp()
    main_window.show()
    sys.exit(app.exec_())

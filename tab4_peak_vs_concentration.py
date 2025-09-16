from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QPushButton,QScrollArea,
    QHBoxLayout, QSpinBox, QLabel, QTextEdit, QLineEdit, QGridLayout, QFileDialog, QComboBox
)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QSplitter
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import pyqtSignal
from staticalclass import PeakAnalysis2
from ramanclass import BarPlotGenerator
from ramanclassauc import BarPlotGenerator1
import pandas as pd
from meanclass import ClusterStatisticsProcessor

class Tab_4:
    def init_tab4(self, parent):
        # Create the Tab 4 widget
        self.tab4 = QWidget(parent)

        # Main layout for the tab
        main_layout = QVBoxLayout(self.tab4)

        # Splitter to separate the two graph areas
        self.splitter = QSplitter(Qt.Horizontal)

        # Left pane for Plot1 (Mean values)
        self.figure_tab4_left = plt.figure()
        self.canvas_tab4_left = FigureCanvas(self.figure_tab4_left)
        self.canvas_tab4_left.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar_tab4_left = NavigationToolbar(self.canvas_tab4_left, self.tab4)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.toolbar_tab4_left)  # Toolbar for the left graph
        left_layout.addWidget(self.canvas_tab4_left)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        self.splitter.addWidget(left_widget)

        # Right pane for Plot2 (AUC values)
        self.figure_tab4_right = plt.figure()
        self.canvas_tab4_right = FigureCanvas(self.figure_tab4_right)
        self.canvas_tab4_right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar_tab4_right = NavigationToolbar(self.canvas_tab4_right, self.tab4)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.toolbar_tab4_right)  # Toolbar for the right graph
        right_layout.addWidget(self.canvas_tab4_right)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        self.splitter.addWidget(right_widget)

        # Adjust splitter behavior
        self.splitter.setStretchFactor(0, 1)  # Left pane resizes proportionally
        self.splitter.setStretchFactor(1, 1)  # Right pane resizes proportionally

        # Add splitter to main layout
        main_layout.addWidget(self.splitter)

        # Graph navigation buttons for Tab 4
        graph_nav_layout4 = QHBoxLayout()

        # Previous and Next navigation buttons
        self.prev_button_tab4 = QPushButton("Previous", self.tab4)
        self.prev_button_tab4.clicked.connect(self.show_previous_graph_tab4)
        self.next_button_tab4 = QPushButton("Next", self.tab4)
        self.next_button_tab4.clicked.connect(self.show_next_graph_tab4)

        graph_nav_layout4.addWidget(self.prev_button_tab4)
        graph_nav_layout4.addWidget(self.next_button_tab4)
        graph_nav_layout4.addStretch()


        # Add graph navigation layout to the main layout
        main_layout.addLayout(graph_nav_layout4)

        # Error widget for Tab 4
        self.error_widget_tab4 = QTextEdit(self.tab4)
        self.error_widget_tab4.setReadOnly(True)
        self.error_widget_tab4.setStyleSheet("color: red;")
        self.error_widget_tab4.setPlaceholderText("Errors and messages will appear here.")
        self.error_widget_tab4.setMaximumHeight(100)  # Set a fixed maximum height
        self.error_widget_tab4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Add the error widget to the main layout
        main_layout.addWidget(self.error_widget_tab4)

        # Add the tab to the parent tab widget
        if isinstance(parent, QTabWidget):
            parent.addTab(self.tab4, "Mean/AUC")


    def set_file_path(self, file_path):
        """
        Set the file path for the process button in Tab 4.
        """
        self.file_path = file_path
        self.error_widget_tab4.append(f"File path set to: {file_path}")

    def show_previous_graph_tab4(self):
        """Show the previous graph in both the left and right panes."""
        if hasattr(self, 'graphs_left1') and self.graphs_left1 and self.current_graph_index > 0:
            self.current_graph_index -= 1
            self.show_current_graph_tab4()

    def show_next_graph_tab4(self):
        """Show the next graph in both the left and right panes."""
        if hasattr(self, 'graphs_left1') and self.graphs_left1 and self.current_graph_index < len(self.graphs_left1) - 1:
            self.current_graph_index += 1
            self.show_current_graph_tab4()

    def show_current_graph_tab4(self):
        """Display the current graph in both the left and right panes."""
        if hasattr(self, 'graphs_left1') and self.graphs_left1:
            fig_left = self.graphs_left1[self.current_graph_index]
            self.canvas_tab4_left.figure = fig_left
            self.canvas_tab4_left.draw()

        if hasattr(self, 'graphs_right1') and self.graphs_right1:
            fig_right = self.graphs_right1[self.current_graph_index]
            self.canvas_tab4_right.figure = fig_right
            self.canvas_tab4_right.draw()

    
        
    def PlotBoth2(self, processed_output_path):
        """Generate and display both Plot1 and Plot2 simultaneously."""
        try:
            
            # Load the statistics output file
            stats_file = processed_output_path
            df_stats = pd.read_excel(stats_file)

            # Ensure the data is not empty
            if df_stats.empty:
                self.error_widget_tab4.setText("Statistics file is empty. Ensure the process ran correctly.")
                print("Statistics file is empty.")
                return

            # Proceed with the plotting logic
            print(f"Loaded statistics file: {stats_file}")
            self.error_widget_tab4.append(f"Statistics file loaded successfully: {stats_file}")

            # Your plotting logic will follow here...
        except Exception as e:
            self.error_widget_tab4.append(f"Error loading statistics file: {str(e)}")
            print(f"Error loading statistics file: {str(e)}")
        try:
            # Clear the error widget before starting
            self.error_widget_tab4.clear()
            print("Cleared error widget.")

            # Prepare mean values (Plot1)
            plotter1 = BarPlotGenerator(processed_output_path)
            # Prepare AUC values (Plot2)
            plotter2 = BarPlotGenerator1(processed_output_path)
            # Initialize graph storage for navigation
            self.graphs_left1 = []  # To store mean value figures
            self.graphs_right1 = []  # To store AUC value figures
            
            unique_wavenumbers1 = plotter1.df_peaks4['mean_wavenumber'].unique()
            unique_wavenumbers2 = plotter2.df_peaks1['mean_wavenumber'].unique()
            
            for wavenumber in unique_wavenumbers1:
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    subset_df = plotter1.df_peaks4[plotter1.df_peaks4['mean_wavenumber'] == wavenumber]
                    plotter1._plot_bar_and_regression(subset_df, ax, wavenumber)
                    self.graphs_left1.append(fig)
                except Exception as e:
                    print(f"Skipping wavenumber {wavenumber} due to error: {e}")
                    continue

            for wavenumber1 in unique_wavenumbers2:
                try:
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    subset_df1 = plotter2.df_peaks1[plotter2.df_peaks1['mean_wavenumber'] == wavenumber1]
                    plotter2._plot_bar_and_regression1(subset_df1, ax1, wavenumber1)
                    self.graphs_right1.append(fig1)
                except Exception as e:
                    print(f"Skipping wavenumber {wavenumber1} due to error: {e}")
                    continue

            # Display the first graph in both panes
            if self.graphs_left1 and self.graphs_right1:
                self.current_graph_index = 0
                self.show_current_graph_tab4()

            # Provide feedback on successful execution
            self.error_widget_tab4.setText(f"Plots generated successfully.")
            print(f"Plots generated successfully.")

        except Exception as e:
            # Handle and display any errors during plotting
            self.error_widget_tab4.setText(f"An error occurred during plotting: {str(e)}")
            print(f"An error occurred during plotting: {str(e)}")

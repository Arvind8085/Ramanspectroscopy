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
from R2singleclass import R2Analysis
from slopeclass import SlopeAnalysis

class Tab_3:
    def init_tab3(self, parent):
        # Create the Tab 3 widget
        self.tab3 = QWidget(parent)

        # Main layout for the tab
        main_layout = QVBoxLayout(self.tab3)

        # Splitter to separate the two graph areas
        self.splitter = QSplitter(Qt.Horizontal)

        # Left pane for Plot1 (R² values)
        self.figure_tab3_left = plt.figure()
        self.canvas_tab3_left = FigureCanvas(self.figure_tab3_left)
        self.canvas_tab3_left.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar_tab3_left = NavigationToolbar(self.canvas_tab3_left, self.tab3)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.toolbar_tab3_left)
        left_layout.addWidget(self.canvas_tab3_left)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        self.splitter.addWidget(left_widget)

        # Right pane for Plot2 (Slope values)
        self.figure_tab3_right = plt.figure()
        self.canvas_tab3_right = FigureCanvas(self.figure_tab3_right)
        self.canvas_tab3_right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar_tab3_right = NavigationToolbar(self.canvas_tab3_right, self.tab3)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.toolbar_tab3_right)
        right_layout.addWidget(self.canvas_tab3_right)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        self.splitter.addWidget(right_widget)

        # Adjust splitter behavior
        self.splitter.setStretchFactor(0, 1)  # Left pane resizes proportionally
        self.splitter.setStretchFactor(1, 1)  # Right pane resizes proportionally

        # Add splitter to main layout
        main_layout.addWidget(self.splitter)

        # Graph navigation buttons for Tab 3
        graph_nav_layout3 = QHBoxLayout()

        # Previous and Next navigation buttons
        self.prev_button_tab3 = QPushButton("Previous", self.tab3)
        self.prev_button_tab3.clicked.connect(self.show_previous_graph_tab3)
        self.next_button_tab3 = QPushButton("Next", self.tab3)
        self.next_button_tab3.clicked.connect(self.show_next_graph_tab3)

        graph_nav_layout3.addWidget(self.prev_button_tab3)
        graph_nav_layout3.addWidget(self.next_button_tab3)
        graph_nav_layout3.addStretch()

        # Single Plot button to generate both plots
        self.plot_button = QPushButton("Generate Plots", self.tab3)
        self.plot_button.clicked.connect(lambda: self.PlotBoth(top_n=60))
        graph_nav_layout3.addWidget(self.plot_button)

        # Add graph navigation layout to the main layout
        main_layout.addLayout(graph_nav_layout3)

        # Error widget for Tab 3
        self.error_widget_tab3 = QTextEdit(self.tab3)
        self.error_widget_tab3.setReadOnly(True)
        self.error_widget_tab3.setStyleSheet("color: red;")
        self.error_widget_tab3.setPlaceholderText("Errors and messages will appear here.")
        self.error_widget_tab3.setMaximumHeight(100)  # Set a fixed maximum height
        self.error_widget_tab3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Add error widget to the main layout
        main_layout.addWidget(self.error_widget_tab3)

        # Add the tab to the parent tab widget
        if isinstance(parent, QTabWidget):
            parent.addTab(self.tab3, "R²/Slopes")


    def set_analysis_output_path(self, path):
        """
        Setter method to receive the analysis output path from the Main GUI.
        """
        self.analysis_output_path = path
        print(f"Analysis output path set to: {path}")
        self.error_widget_tab3.append(f"Analysis path set to: {path}")

    def show_previous_graph_tab3(self):
        """Show the previous graph in both the left and right panes."""
        if hasattr(self, 'graphs_left') and self.graphs_left and self.current_graph_index > 0:
            self.current_graph_index -= 1
            self.show_current_graph_tab3()

    def show_next_graph_tab3(self):
        """Show the next graph in both the left and right panes."""
        if hasattr(self, 'graphs_left') and self.graphs_left and self.current_graph_index < len(self.graphs_left) - 1:
            self.current_graph_index += 1
            self.show_current_graph_tab3()

    def show_current_graph_tab3(self):
        """Display the current graph in both the left and right panes."""
        if hasattr(self, 'graphs_left') and self.graphs_left:
            fig_left = self.graphs_left[self.current_graph_index]
            self.canvas_tab3_left.figure = fig_left
            self.canvas_tab3_left.draw()

        if hasattr(self, 'graphs_right') and self.graphs_right:
            fig_right = self.graphs_right[self.current_graph_index]
            self.canvas_tab3_right.figure = fig_right
            self.canvas_tab3_right.draw()

    def PlotBoth(self, top_n=60):
    
        """Generate and display both Plot1 and Plot2 simultaneously."""
        if not hasattr(self, 'analysis_output_path') or not self.analysis_output_path:
            self.error_widget_tab3.setText("No analysis file found. Run the analysis first.")
            print("No analysis file found.")
            return

        try:
            # Clear the error widget before starting
            self.error_widget_tab3.clear()
            print("Cleared error widget.")

            # Prepare R² values (Plot1)
            plotter1 = R2Analysis(self.analysis_output_path, output_dir=None)
            plotter1.load_and_process_data()
            plotter1.generate_r2_collection()

            # Prepare Slope values (Plot2)
            plotter2 = SlopeAnalysis(self.analysis_output_path, output_dirs=None)
            plotter2.load_and_prepare_data()
            plotter2.process_all_priorities()

            # Initialize graph storage for navigation
            self.graphs_left = []  # To store R² value figures
            self.graphs_right = []  # To store Slope value figures

            # Prepare Plot1 (R² values)
            priorities1 = list(plotter1.r2_collection.keys())[:top_n]
            for priority in priorities1:
                r2_values = plotter1.r2_collection[priority]
                if not r2_values:
                    continue

                # Filter the dataframe for the current priority
                priority_df = plotter1.analysis_df[plotter1.analysis_df['Priority'] == priority]

            
                # Extract unique cluster ratios for the current priority
                cluster_ratios = priority_df[['Cluster 1', 'Cluster 2']].drop_duplicates()
                cluster_ratio_str = ", ".join([f"{row['Cluster 1']}/{row['Cluster 2']}" for _, row in cluster_ratios.iterrows()])

                fig, ax = plt.subplots(figsize=(10, 6))
                subset_labels = [f"Subset {k+1} ({len(r2_values) + 2 - k} conc.)" for k in range(len(r2_values))]

                ax.bar(subset_labels, r2_values, color='skyblue', width=0.1)
                ax.set_xlabel('Subsets')
                ax.set_ylabel('R² Values')

                # Set title with Clusters and Mean Wavenumbers
                ax.set_title(f"Clusters: {cluster_ratio_str} - Mean Wavenumbers: {priority_df['Wavenumber 1'].mean():.2f}/{priority_df['Wavenumber 2'].mean():.2f} - Priority {priority}")


                # Set legend location
                ax.legend(['R² Values'], loc='upper left')

                ax.set_xticklabels(subset_labels, rotation=45)
                self.graphs_left.append(fig)


        

            # Prepare Plot2 (Slope values)
            priorities2 = list(plotter2.slope_collection.keys())[:top_n]
            for priority in priorities2:
                slope_values = plotter2.slope_collection[priority]
                 # Extract unique cluster ratios for the current priority
               
                if not slope_values:
                    continue
                # Filter the dataframe for the current priority
                priority_df1 = plotter2.analysis_dfs[plotter2.analysis_dfs['Priority'] == priority]
                cluster_ratios2 = priority_df1[['Cluster 1', 'Cluster 2']].drop_duplicates()
                cluster_ratio_str2 = ", ".join([f"{row['Cluster 1']}/{row['Cluster 2']}" for _, row in cluster_ratios2.iterrows()])
                fig, ax = plt.subplots(figsize=(10, 6))
                subset_labels = [f"Subset {k+1} ({len(slope_values) + 2 - k} conc.)" for k in range(len(slope_values))]
                ax.plot(subset_labels, slope_values, marker='o', linestyle='-', color='blue')
                ax.set_xlabel('Subsets')
                ax.set_ylabel('Slope Values')
                # Set title with Clusters and Mean Wavenumbers
                ax.set_title(f"Clusters: {cluster_ratio_str2} - Mean Wavenumbers: {priority_df1['Wavenumber 1'].mean():.2f}/{priority_df1['Wavenumber 2'].mean():.2f} - Priority {priority}")
                # Set legend location
                ax.legend(['slope Values'], loc='upper left')
                ax.set_xticklabels(subset_labels, rotation=45)
                self.graphs_right.append(fig)

            # Display the first graph in both panes
            if self.graphs_left and self.graphs_right:
                self.current_graph_index = 0
                self.show_current_graph_tab3()

            # Provide feedback on successful execution
            self.error_widget_tab3.setText(f"Plots generated successfully (Top {top_n}).")
            print(f"Plots generated successfully (Top {top_n}).")
        except Exception as e:
            # Handle and display any errors during plotting
            self.error_widget_tab3.setText(f"An error occurred during plotting: {str(e)}")
            print(f"An error occurred during plotting: {str(e)}")

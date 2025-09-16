from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit, QTabWidget, QFileDialog, QSizePolicy
from preprocessclass1 import ExcelProcessor1
from preprocessclass2 import EnhancedExcelSheetProcessor
from concentration_class import ExcelConcentrationMapper
from PyQt5.QtCore import Qt

class Preprocess:
    def __init__(self):
        self.input_file_path = None  # To store the path of the input file

    def init_tab0(self, parent):
        # Create the Preprocess tab widget
        self.tab_preprocess = QWidget(parent)

        # Main layout for the tab
        main_layout = QVBoxLayout(self.tab_preprocess)

        # Container layout for buttons
        button_layout = QVBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)  # Center-align the buttons

        # Set button sizes
        button_width = 150
        button_height = 40

        # Load button
        self.load_button = QPushButton("Load File", self.tab_preprocess)
        self.load_button.setFixedSize(button_width, button_height)
        self.load_button.clicked.connect(self.load_file)
        button_layout.addWidget(self.load_button)

        # Preprocess button
        self.preprocess_button = QPushButton("Preprocess", self.tab_preprocess)
        self.preprocess_button.setFixedSize(button_width, button_height)
        self.preprocess_button.clicked.connect(self.preprocess)
        button_layout.addWidget(self.preprocess_button)

        # Assign CCDLoad button
        self.run_button = QPushButton("Assign CCDload", self.tab_preprocess)
        self.run_button.setFixedSize(button_width, button_height)
        self.run_button.clicked.connect(self.assign)
        button_layout.addWidget(self.run_button)

        # Assign Concentrations button
        self.assign_concentrations_button = QPushButton("Assign Concentrations", self.tab_preprocess)
        self.assign_concentrations_button.setFixedSize(button_width, button_height)
        self.assign_concentrations_button.clicked.connect(self.assign_concentrations)
        button_layout.addWidget(self.assign_concentrations_button)

        # Add the button layout to the main layout
        main_layout.addLayout(button_layout)

        # Error widget for displaying messages or errors
        self.error_widget = QTextEdit(self.tab_preprocess)
        self.error_widget.setReadOnly(True)
        self.error_widget.setStyleSheet("color: red;")
        self.error_widget.setPlaceholderText("Errors and messages will appear here.")
        self.error_widget.setMaximumHeight(100)  # Set a fixed maximum height
        self.error_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Add the error widget to the main layout
        main_layout.addWidget(self.error_widget)

        # Add the tab to the parent tab widget
        if isinstance(parent, QTabWidget):
            parent.addTab(self.tab_preprocess, "Preprocess")


    def load_file(self):
        """Load an Excel file using a file dialog."""
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(None, "Select Excel File", "", "Excel Files (*.xlsx);;All Files (*)", options=options)
        if file_path:
            self.input_file_path = file_path
            self.error_widget.append(f"File loaded: {file_path}")

    def preprocess(self):
        if not self.input_file_path:
            self.error_widget.append("Error: No file loaded. Please load a file first.")
            return
        self.error_widget.append("Preprocessing started...")
        try:
            # Example usage of ExcelProcessor1
            processor1 = ExcelProcessor1(self.input_file_path)
            processor1.run()
            self.error_widget.append("Preprocessing completed.")
        except Exception as e:
            self.error_widget.append(f"Error during preprocessing: {e}")

    def assign(self):
        if not self.input_file_path:
            self.error_widget.append("Error: No file loaded. Please load a file first.")
            return
        self.error_widget.append("Assigning CCDLoad started...")
        try:
            # Example usage of ExcelSheetProcessor3
            processor3 = EnhancedExcelSheetProcessor(self.input_file_path)
            processor3.run()
            self.error_widget.append("Assigning CCDLoad completed.")
        except Exception as e:
            self.error_widget.append(f"Error during CCDLoad assignment: {e}")

    def assign_concentrations(self):
       
        if not self.input_file_path:
            self.error_widget.append("Error: No file loaded. Please load a file first.")
            return
        self.error_widget.append("Assigning concentrations started...")
        try:
            # Using ExcelConcentrationMapper to process the file
            concentration_mapper = ExcelConcentrationMapper(self.input_file_path, error_widget=self.error_widget)
            concentration_mapper.process()
            self.error_widget.append("Assigning concentrations completed successfully.")
        except Exception as e:
            self.error_widget.append(f"Error during concentration assignment: {e}")


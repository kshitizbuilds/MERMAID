# 📊 Data Visualization and Analysis App

This project leverages **Streamlit** to create an interactive web application for data processing and visualization, primarily using **Pandas** for data manipulation and **Graphviz** for generating graphs or diagrams.

---

## 🚀 Getting Started

Follow these steps to set up and run the application on your local machine.

### 📋 Prerequisites

You must have **Python 3.x** installed. Additionally, this project requires the external **Graphviz** software to be installed on your operating system.

### 1. Install Graphviz (External Software)

The Python `graphviz` package is a wrapper; it requires the core Graphviz software to be installed to function correctly.

Please download and install it from the official website:
<https://graphviz.org/download/>

### 2. Install Python Dependencies

Install all the necessary Python packages using `pip`:

```bash
pip install pandas streamlit graphviz openpyxl
Package	Role in Project
pandas	Core library for data manipulation and analysis.
streamlit	Used to build the interactive web application.
graphviz	Enables graph and diagram visualization.
openpyxl	Used for reading and writing Excel (.xlsx) files.

Export to Sheets
🏃 Running the Application
After installing all the prerequisites, run the Streamlit app from your terminal.

Ensure your main application file, app.py, is in your current directory.
Ensure a folder named Graphviz is present in root directory of the project.

Execute the following command:

Bash

streamlit run app.py
The application will automatically open in your default browser at http://localhost:8501.

✨ Summary of Needed Commands
Bash

# 1. Install required packages
pip install pandas streamlit graphviz openpyxl

# 2. Run the application
streamlit run app.py
```

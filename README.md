# Metagenomic Analysis of Mixed Anaerobic Microbiomes
## Overview
This repository contains analyses of two datasets composed of metagenomes from microbiomes. The project focuses on selecting relevant bioindicators for predicting quantitative metrics involved in biogas production assessment. Exploratory analyses will be included to understand the microbial diversity in the samples along with the interpretation of trained classification models.

## Repository Structure
- notebooks/: Jupyter notebooks containing the analyses.
- scripts/: Scripts used in the analyses.
- outputs/: Results from the analyses.
-docs/: Additional documentation.

## Setup
1. Clone this repository:
    ```bash
    https://github.com/Anajulia-gon/Bettle-data-analysis.git
    ```
2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
## Execution
- **Data Collection**: The datasets are stored on a private server, consisting of separate FASTA files averaging 20.8475GB and 16.4627GB, sourced from Data Sources 1 and 2, respectively. The total size is 2,125.63GB.
- **Data Preprocessing**: Using the MuDoGeR tool, sequences were cleaned and organized, while biologically significant information was extracted and summarized into tables.
- **Data Loading and Transformation**: Relative abundance tables of OTUs were loaded into DataFrames for further manipulation. Typing and grouping were performed, and metadata tables were merged to create categorical variable columns.
- **Exploratory Data Analysis (EDA)**: Alpha and Beta diversity of the communities were explored, along with the visualization of categorical data.
- **Data Modeling**: The data was prepared for machine learning pipelines.
- **Model Evaluation**: Model performance was assessed using G-mean and GridSearch.
- Model Interpretation: SHAP plots were generated for model interpretation.

## Main Results
- Microbial Diversity: Key findings on the microbial diversity associated with the Sun beetle.
- Microbial Functions: Initial analysis of the role played by the main attributes of the microbial communities.

## Contact information
Autor: Ana Julia  
Email: ana.tendulini@gmail.com

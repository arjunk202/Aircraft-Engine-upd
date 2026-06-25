Aircraft Engine Predictive Maintenance (PM)
Overview
This project builds a Predictive Maintenance pipeline for aircraft engines using sensor/maintenance data.
The goal is to predict engine health / failure risk (or remaining useful life–style targets, depending on the dataset labels) so maintenance can be planned proactively.

Repository Structure
	* Data_Report_Artifacts/: Any generated reports/artifacts (plots, metrics, processed outputs).
	* Materials/: Supporting docs/resources.
	* PM_train.csv: Training dataset.
	* PM_test.csv: Test dataset.
	* PM_truth.csv: Ground-truth labels/targets used for evaluation.
	* data_cleaning_notebook.ipynb: Data cleaning and preprocessing steps.
	* train_stage1.csv / val_stage1.csv / test_stage1.csv: Stage-1 split datasets (for model training/validation/testing).

Project Workflow
	1. Data Cleaning & Preprocessing

		* Performed in: data_cleaning_notebook.ipynb
		* Includes cleaning missing values, feature engineering/selection, and final preparation for modeling.



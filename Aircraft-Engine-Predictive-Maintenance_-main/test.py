# Import all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
df_train = pd.read_csv('pm_train.csv')
df_test = pd.read_csv('pm_test.csv')
df_truth = pd.read_csv('pm_truth.csv')

print("✅ All files loaded successfully!")
print("Training data shape:", df_train.shape)
print("Test data shape:", df_test.shape)
print("Truth data shape:", df_truth.shape)
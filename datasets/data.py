import pandas as pd

# Load the dataset
inventory_df = pd.read_csv('datasets/inventory_updated.csv')
recipes_df = pd.read_csv("datasets/recipes.csv", engine="python", on_bad_lines="skip")
sales_df = pd.read_csv("datasets/prod_sales.csv")

# Create emoty cart dictionary
carts = {}
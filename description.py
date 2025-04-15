import pandas as pd

# Load the dataset
df = pd.read_csv("facemesh_expression_data.csv")

# 1. Show the first few rows of the dataset to get a quick look at the data
print("First few rows of the dataset:")
print(df.head())

# 2. Show the shape of the dataset (rows, columns)
print("\nShape of the dataset (rows, columns):")
print(df.shape)

# 3. Display all column names
print("\nColumns in the dataset:")
print(df.columns)

# 4. Summary statistics for numerical columns
print("\nSummary statistics of the dataset:")
print(df.describe())

# 5. Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# 6. Display the data types of each column
print("\nData types of each column:")
print(df.dtypes)

# 7. Count the number of face points (features) excluding the label column
num_face_points = df.drop('label', axis=1).shape[1]
print(f"\nNumber of face points (features): {num_face_points}")

# 8. If you want to check the unique expressions (labels) in your dataset
print("\nUnique expressions (labels) in the dataset:")
print(df['label'].unique())

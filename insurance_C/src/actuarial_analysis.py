import pandas as pd

# Load the dataset
df = pd.read_excel('insurance\data\Portfolio.xlsx')

# Converting the date columns to datetime
df['Birth_Date'] = pd.to_datetime(df['Birth_Date'], format='%d/%m/%Y')
df['Effective_Date'] = pd.to_datetime(df['Effective_Date'], format='%d/%m/%Y')
df['Renewal_Date'] = pd.to_datetime(df['Renewal_Date'], format='%d/%m/%Y')
df['Birthday'] = pd.to_datetime(df['Birthday'], format='%d/%m/%Y')

# Check for missing values
print(df.isnull().sum())

# Drop missing values
df = df.dropna()

# Summary statistics
print(df.describe())

import seaborn as sns
import matplotlib.pyplot as plt

# Visualize distributions
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution with KDE')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# BoxPlot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Capital', data=df)
plt.title('Capital Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Capital')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include='number')  # Select only numeric columns
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numeric Variables')
plt.show()

# Calculate duration of policy in years
df['Policy_Duration'] = (df['Renewal_Date'] - df['Effective_Date']).dt.days / 365.25

# Convert categorical variables to numeric
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# Display the first few rows after transformation
print(df.head())

import scipy.stats as stats

# T-test for Capital difference by Gender
male_capital = df[df['Gender_M'] == 1]['Capital']
female_capital = df[df['Gender_M'] == 0]['Capital']
t_stat, p_value = stats.ttest_ind(male_capital, female_capital)
print(f'T-statistic: {t_stat}, P-value: {p_value}')


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define features and target variable
X = df[['Age', 'Policy_Duration', 'Gender_M', 'Age_Actuarial', 'Age_actuarial_quarter', 'Month']]
y = df['Capital']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}, R-squared: {r2}')

import matplotlib.pyplot as plt

# Visualizing the impact of Age on Capital
plt.scatter(df['Age'], df['Capital'], c='blue')
plt.xlabel('Age')
plt.ylabel('Capital')
plt.title('Age vs Capital')
plt.show()



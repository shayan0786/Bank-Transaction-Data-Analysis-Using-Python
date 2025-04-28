import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
df = pd.read_csv("bank_transactions_data.csv")
# Q1. Display the first 10 rows of the dataset
print(df.head(10))
# Q2. Check the data types of each column
print(df.dtypes)
# Q3. Convert TransactionDate and PreviousTransactionDate to datetime objects
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
df["PreviousTransactionDate"] = pd.to_datetime(df["PreviousTransactionDate"])
# Q4. Find the total number of unique customers (based on AccountID)
unique_customers = df["AccountID"].nunique()
print("Total unique customers:", unique_customers)
# Q5. Show the number of transactions per TransactionType
df["TransactionType"].value_counts()
print("Number of transactions per TransactionType:")
print(df["TransactionType"].value_counts())
#  Q6. Sort the dataset by TransactionAmount in descending order and show top 5
df.sort_values(by="TransactionAmount", ascending=False).head(5)
print("Top 5 transactions by amount:")
print(df.sort_values(by="TransactionAmount", ascending=False).head(5))
#Part 2
# Q1: Check for Missing Values in the Dataset
# Count missing values column-wise
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)
# Q2: Display All Rows That Have Any Missing Value
rows_with_missing = df[df.isnull().any(axis=1)]
print(rows_with_missing)
# Q3: Drop All Rows Containing Any Missing Values
df_dropped = df.dropna()
print("Shape after dropping rows with NaN:", df_dropped.shape)
# Q4: Fill Missing Values in a Specific Column with Mean/Mode/Median
median_value = df['TransactionAmount'].median()
df.fillna({'TransactionAmount': median_value}, inplace=True)
# Q5: Replace Specific Placeholder Strings Like ‘N/A’, ‘Missing’ etc.
df.replace(['N/A', 'n/a', 'missing', 'Missing'], pd.NA, inplace=True)
# Q6: Check for Duplicate Rows and Remove Them
duplicates = df.duplicated().sum()
print("Total duplicate rows:", duplicates)
df_cleaned = df.drop_duplicates()
print("Shape after removing duplicates:", df_cleaned.shape)
# Q7 (Mini EDA Touch): Check for Outliers Using IQR in ‘TransactionAmount'
# "Detect possible outliers in 'TransactionAmount' using the IQR method."
Q1 = df['TransactionAmount'].quantile(0.25)
Q3 = df['TransactionAmount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['TransactionAmount'] < lower_bound) | (df['TransactionAmount'] > upper_bound)]
print("Number of outliers:", outliers.shape[0])
#Part 3
# Q1. Bar Chart using Matplotlib: Visualize Number of Transactions per Transaction Type
type_counts = df['TransactionType'].value_counts()
colors = ['skyblue', 'lightgreen'] * len(type_counts)
plt.figure(figsize=(10,6))
bars = plt.bar(
    type_counts.index, 
    type_counts.values, 
    color=colors[:len(type_counts)],  
    edgecolor='black'
)

plt.title('Number of Transactions per Transaction Type', fontsize=14)
plt.xlabel('Transaction Type', fontsize=12)
plt.ylabel('Number of Transactions', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# Q2. Plot the distribution of transaction amounts using a histogram.
plt.figure(figsize=(10, 6))
plt.hist(df['TransactionAmount'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
# Q3. Visualize the distribution of Transaction Amounts across different Customer Occupations using a Boxplot.
plt.figure(figsize=(12, 6))
sns.boxplot(x='CustomerOccupation', y='TransactionAmount', data=df, palette='Set3')
plt.title('Transaction Amount by Customer Occupation', fontsize=14, fontweight='bold')
plt.xlabel('Customer Occupation', fontsize=12)
plt.ylabel('Transaction Amount', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# Q4. Create a countplot showing the number of transactions for each transaction type.
plt.figure(figsize=(10, 6))
sns.countplot(x='TransactionType', data=df, palette='viridis')
plt.title('Transaction Count by Type')
plt.xlabel('Transaction Type')
plt.ylabel('Number of Transactions')
plt.grid(True)
plt.show()
# Q5. Show a pie chart of the proportion of transaction types.
plt.figure(figsize=(8, 8))
df['TransactionType'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Transaction Type Distribution')
plt.ylabel('')
plt.show()
# Q6. Visualize the trend of transactions over time using a line plot.
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
daily_trend = df.groupby(df['TransactionDate'].dt.date)['TransactionAmount'].sum()
plt.figure(figsize=(12, 6))
plt.plot(daily_trend.index, daily_trend.values, color='teal')
plt.title('Daily Transaction Trend')
plt.xlabel('Date')
plt.ylabel('Total Transaction Amount')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
# Q7. Use a heatmap to show correlation between numeric columns.
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
# Q8. Visualize the distribution of Transaction Amounts for each Transaction Type using a Violin Plot.
plt.figure(figsize=(10, 6))
sns.violinplot(x='TransactionType', y='TransactionAmount', data=df, palette='muted')
plt.title('Transaction Amount by Transaction Type (Violin Plot)', fontsize=14, fontweight='bold')
plt.xlabel('Transaction Type', fontsize=12)
plt.ylabel('Transaction Amount', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
avg_amount = df.groupby('AccountType')['TransactionAmount'].mean().sort_values()
plt.figure(figsize=(10, 6))
avg_amount.plot(kind='barh', color='coral')
plt.title('Average Transaction Amount by Account Type')
plt.xlabel('Average Amount')
plt.ylabel('Account Type')
plt.grid(True)
plt.show()
# Q9. Find the average Transaction Amount for each Transaction Type and visualize using a horizontal bar chart.
avg_amount = df.groupby('TransactionType')['TransactionAmount'].mean().sort_values()
# Plot as horizontal bar chart
plt.figure(figsize=(10, 6))
avg_amount.plot(kind='barh', color='coral')
plt.xlabel('Average Transaction Amount')
plt.title('Average Transaction Amount by Transaction Type', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
# Q10. Plot KDE (Kernel Density Estimation) of transaction amounts.
plt.figure(figsize=(10, 6))
sns.kdeplot(df['TransactionAmount'], shade=True, color='purple')
plt.title('KDE Plot of Transaction Amounts')
plt.xlabel('Transaction Amount')
plt.grid(True)
plt.show()
# ✅ Q8. Is there a relationship between login attempts and transaction amount?
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='LoginAttempts', y='TransactionAmount', color='mediumvioletred')
plt.title('Login Attempts vs Transaction Amount')
plt.xlabel('Login Attempts')
plt.ylabel('Transaction Amount')
plt.grid(True)
plt.tight_layout()
plt.show()
#Part 4
mean_amount = df['TransactionAmount'].mean()
median_amount = df['TransactionAmount'].median()
mode_amount = df['TransactionAmount'].mode()[0]  # sabse pehla mode

print("Average Transaction Amount:", mean_amount)
print("Median Transaction Amount:", median_amount)
print("Mode Transaction Amount:", mode_amount)


# %%
# Q2. What is the standard deviation and variance of transaction durations?
std_duration = df['TransactionDuration'].std()
var_duration = df['TransactionDuration'].var()
print("Standard Deviation of Duration:", std_duration)
print("Variance of Duration:", var_duration)
# Q3. Is there a correlation between TransactionAmount and AccountBalance?
correlation = df['TransactionAmount'].corr(df['AccountBalance'])
print("Correlation between TransactionAmount and AccountBalance:", correlation)
# Q4. How are numerical features correlated with each other?
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
plt.show()
# Q5. Which channel shows the most variability in transaction amount?
channel_variance = df.groupby('Channel')['TransactionAmount'].var().sort_values(ascending=False)
print("Transaction Amount Variance by Channel:\n", channel_variance)
#  Q6. Are there any outliers in TransactionAmount?
sns.boxplot(x=df['TransactionAmount'], color='orange')
plt.title("Boxplot of Transaction Amount")
plt.show()
# Q7. Probability distribution of TransactionAmount — is it Normal?
sns.histplot(df['TransactionAmount'], kde=True, bins=30, color='teal')
plt.title("Transaction Amount Distribution")
plt.show()






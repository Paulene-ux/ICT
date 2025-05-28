1:
#seaborn

dx = pd.DataFrame({'CRC': b, 'HSH': c, 'Height': d})
corr = dx.corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Heatmap')
plt.show()
   2:
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

#DATA SET
df = pd.read_csv(r'C:\Users\hmcldryl\Desktop\cupsnheight.csv')

a = df['Person'].to_numpy()
b = df['Cups of Rice Consumed'].to_numpy()
c = df['Hours Spent on Homework'].to_numpy()
d = df['Height'].to_numpy()


#INITIALS
col_title = df.columns.tolist()
col_cat = ['Person', 'Cups of Rice Consumed', 'Hours Spent on Homework', 'Height']

#OUTPUTS
print("Synthetic Dataset of Dietary, Academic, and Physical Attributes\n")

print("Data set columns:")
for i in range(len(col_cat)):
    print(i + 1, col_cat[i], col_title[i])
    
#proof of clean data set
print("\nOriginal Data Shape:", df.shape)
print("Missing Values:") 
print(df.isnull().sum())
print("\nTop 5 Person:\n", df.head())
   3: %history -n 1-505
   4: %history -n 1-505
   5:
#numpy operations

sumcrc = np.sum(b)
maxcrc = np.max(b)
mincrc = np.min(b)
meanhsh = np.mean(c)
medianhsh = np.median(c)
perhsh = np.percentile(c, [45, 50, 55])
stdh = np.std(d)
varh = np.var(d)
rangeh = np.ptp(d)

print("\nCup of Rice Consumed Statistics:")
print(f"Sum of Cup of Rice Consumed: {sumcrc:.1f}")
print(f"Maximum of Cup of Rice Consumed: {maxcrc:.1f}")
print(f"Minimum of Cup of Rice Consumed: {mincrc:.1f}")

print("\nHours Spent on Homework Statistics:")
print(f"Mean of Hours Spent on Homework: {meanhsh:.1f}")
print(f"Median of Hours Spent on Homework: {medianhsh:.1f}")
print(f"45th, 50th, and 55th Percentile of Hours Spent on Homework: {perhsh}")

print("\nHeight Statistics:")
print(f"Standard Deviation of Height: {stdh:.1f}")
print(f"Variance of Height: {varh:.1f}")
print(f"Range of Height: {rangeh:.1f}")
   6:
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

#DATA SET
df = pd.read_csv(r'C:\Users\hmcldryl\Desktop\cupsnheight.csv')

a = df['Person'].to_numpy()
b = df['Cups of Rice Consumed'].to_numpy()
c = df['Hours Spent on Homework'].to_numpy()
d = df['Height'].to_numpy()


#INITIALS
col_title = df.columns.tolist()
col_cat = ['Person', 'Cups of Rice Consumed', 'Hours Spent on Homework', 'Height']

#OUTPUTS
print("Synthetic Dataset of Dietary, Academic, and Physical Attributes\n")

print("Data set columns:")
for i in range(len(col_cat)):
    print(i + 1, col_cat[i], col_title[i])
    
#proof of clean data set
print("\nOriginal Data Shape:", df.shape)
print("Missing Values:") 
print(df.isnull().sum())
print("\nTop 5 Person:\n", df.head())
   7:
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

#DATA SET
df = pd.read_csv(r'C:\Users\hmcldryl\Desktop\cupsnheight.csv')

a = df['Person'].to_numpy()
b = df['Cups of Rice Consumed'].to_numpy()
c = df['Hours Spent on Homework'].to_numpy()
d = df['Height'].to_numpy()


#INITIALS
col_title = df.columns.tolist()
col_cat = ['Person', 'Cups of Rice Consumed', 'Hours Spent on Homework', 'Height']

#OUTPUTS
print("Synthetic Dataset of Dietary, Academic, and Physical Attributes\n")

print("Data set columns:")
for i in range(len(col_cat)):
    print(i + 1, col_cat[i], col_title[i])
    
#proof of clean data set
print("\nOriginal Data Shape:", df.shape)
print("Missing Values:") 
print(df.isnull().sum())
print("\nTop 5 Person:\n", df.head())
   8:
#numpy operations

sumcrc = np.sum(b)
maxcrc = np.max(b)
mincrc = np.min(b)
meanhsh = np.mean(c)
medianhsh = np.median(c)
perhsh = np.percentile(c, [45, 50, 55])
stdh = np.std(d)
varh = np.var(d)
rangeh = np.ptp(d)

print("\nCup of Rice Consumed Statistics:")
print(f"Sum of Cup of Rice Consumed: {sumcrc:.1f}")
print(f"Maximum of Cup of Rice Consumed: {maxcrc:.1f}")
print(f"Minimum of Cup of Rice Consumed: {mincrc:.1f}")

print("\nHours Spent on Homework Statistics:")
print(f"Mean of Hours Spent on Homework: {meanhsh:.1f}")
print(f"Median of Hours Spent on Homework: {medianhsh:.1f}")
print(f"45th, 50th, and 55th Percentile of Hours Spent on Homework: {perhsh}")

print("\nHeight Statistics:")
print(f"Standard Deviation of Height: {stdh:.1f}")
print(f"Variance of Height: {varh:.1f}")
print(f"Range of Height: {rangeh:.1f}")
   9:
#scipy operations

#sampling the data set into 4 arrays (samples) with 5 values each
arr1 = np.array([])
arr2 = np.array([])
arr3 = np.array([])
arr4 = np.array([])
for i in range(20):
    if i < 5:
        arr1 = np.append(arr1, b[i])

    if i < 10 and i > 4:
        arr2 = np.append(arr2, b[i])

    if i < 15 and i > 9:
        arr3 = np.append(arr3, b[i])

    if i < 20 and i > 14:
        arr4 = np.append(arr4, b[i])

print("Array 1", arr1,"\nArray 2", arr2, "\nArray 3",arr3, "\nArray 4",arr4)

#one-way anova
f_statistic, p_value = stats.f_oneway(arr1, arr2, arr3, arr4)
print(f"\nOne-Way Anova \n\nF-statistic: {f_statistic:.4f}, P-value: {p_value:.8f}")

#plot
data = [arr1, arr2, arr3, arr4]
labels = ['Array 1', 'Array 2', 'Array 3', 'Array 4']

plt.figure(figsize=(8, 5))
plt.boxplot(data, tick_labels=labels)
plt.title('One-Way ANOVA: Array Comparisons')
plt.ylabel('Scores')
plt.grid(True)
plt.show()

#standard normal
zval = stats.zscore(arr1)
print(f"\nZ-Distribution\n\nZ-score for first 5 values:")
for i in range(5):
    print(zval[i], end=' ')

#plot
z_range = np.linspace(-3, 3, 50)
pdf = stats.norm.pdf(z_range)

plt.figure(figsize=(10, 4))
plt.plot(z_range, pdf, color='gray', linestyle='-', label='Standard Normal PDF')

plt.hist(zval, bins=20, density=True, alpha=0.5, label='Z-scores', color='darkred')

plt.axvline(0, color='black', linestyle='-', linewidth=3)
plt.title('Z-Score Distribution of Array 1')
plt.xlabel('Z-score')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

#t-test
t_stat, p_value = stats.ttest_ind(arr3, arr4)
print(f"T - Test\n\nT-test between Array 3 and Array 4: \nt = {t_stat:.2f} \np = {p_value:.3f}")

#plot
plt.figure(figsize=(8, 5))
plt.boxplot([arr3, arr4], tick_labels=['Array 3', 'Array 4'])
plt.title('T-Test: Array 3 & Array 4 Comparison')
plt.ylabel('Values')
plt.grid(True)
plt.show()
  10:
#Statsmodels

x = sm.add_constant(df['Hours Spent on Homework'])
model = sm.OLS(df['Cups of Rice Consumed'], x).fit()
print("\nLinear Regression Summary:")
print(model.summary())
  11:
#correlation with visualizations
x = 1
y = 2

#corr1
corr_matrix = np.corrcoef(b, c)
correlation = corr_matrix[0, 1]
print(col_cat[x], col_title[x], "vs", col_cat[y], col_title[y])
print("Correlation coefficient (Pearson r):", correlation)
y += 1

#plot for corr1
ds = pd.DataFrame({'X': b, 'Y': c})

plt.figure(figsize=(8, 5))
sns.regplot(x='X', y='Y', data=ds, color='darkred', marker=',', ci = 95, line_kws={'color': 'red', 'linewidth': 2})

plt.title('Seaborn Scatterplot with Regression Line')
plt.xlabel('Cups of Rice Consumed')
plt.ylabel('Hours Spent on Homework')
plt.grid(True)
plt.show()

#corr2
corr_matrix = np.corrcoef(b, d)
correlation = corr_matrix[0, 1]
print()
print(col_cat[x], col_title[x], "vs", col_cat[y], col_title[y])
print("Correlation coefficient (Pearson r):", correlation)
x += 1

#plot for corr2
ds = pd.DataFrame({'X': b, 'Y': d})

plt.figure(figsize=(8, 5))
sns.regplot(x='X', y='Y', data=ds, color='darkred', marker=',', ci = 95, line_kws={'color': 'red', 'linewidth': 2})

plt.title('Seaborn Scatterplot with Regression Line')
plt.xlabel('Cups of Rice Consumed')
plt.ylabel('Hours Spent on Homework')
plt.grid(True)
plt.show()

#corr3
corr_matrix = np.corrcoef(c, d)
correlation = corr_matrix[0, 1]
print()
print(col_cat[x], col_title[x], "vs", col_cat[y], col_title[y])
print("Correlation coefficient (Pearson r):", correlation)

#plot for corr3
ds = pd.DataFrame({'X': c, 'Y': d})

plt.figure(figsize=(8, 5))
sns.regplot(x='X', y='Y', data=ds, color='darkred', marker=',', ci = 95, line_kws={'color': 'red', 'linewidth': 2})

plt.title('Seaborn Scatterplot with Regression Line')
plt.xlabel('Cups of Rice Consumed')
plt.ylabel('Hours Spent on Homework')
plt.grid(True)
plt.show()
  12:
#seaborn

dx = pd.DataFrame({'CRC': b, 'HSH': c, 'Height': d})
corr = dx.corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Heatmap')
plt.show()
  13: %history -g -f sonio_revisionhistory
  14: %history -g -f sonio_revisionhistory
  15: %history -n 1-505

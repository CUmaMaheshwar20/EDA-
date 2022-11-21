#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[2]:


import numpy as np 
import pandas as pd 


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[5]:


#Reading data 
df = pd.read_csv('cars_raw.csv')


# In[6]:


#Visualizing first 5 rows
df.head()


# In[7]:


#Description of the data in the DataFrame
df.describe()


# In[8]:


#information about the DataFrame
df.info()


# # DATA CLEANING

# Looking into the 'Price' column we can see that it is currently a categorical column. We need to convert it into an integer column. There are some values as 'Not Priced' which needs to be removed. The dollar sign and comma sign in the Price needs to be replaced too.

# PRICE COLUMN

# In[9]:


df = df[df['Price'] != 'Not Priced']


# In[10]:


df['Price'] = df['Price'].str.replace('$','').str.replace(',','')


# In[11]:


df['Price'] = df['Price'].astype(int)


# In[12]:


#Filling the missing values in DealType column with the mode of that column
df['DealType'] = df['DealType'].fillna(df['DealType'].mode()[0])


# In[13]:


df['Transmission'].unique()


# In[14]:


#Removing the rows having data like '-'
df.drop(df[df["Transmission"]=='–'].index, inplace=True)


# In[15]:


def trans(x):
    if 'AUTOMATIC' in x or 'Automatic' in x or 'automatic' in x or 'CVT' in x or 'cvt' in x or 'variable' in x or 'Auto' in x or 'auto' in x or 'A/T' in x:
        return 'Automatic'
    if 'M/T' in x or 'manual' in x or 'Manual' in x or 'Dual Shift' in x or 'Double-clutch' in x or 'PDK' in x:
        return 'Manual'
    else:
        return 'Automatic'


# In[16]:


df['Transmission'] = df['Transmission'].apply(trans)


# In[17]:


df['Engine'].unique()


# In[18]:


#Extracting the fuel capacity of the engine using regular expression
df['Cap'] = df['Engine'].str.findall(r'[0-9].[0-9]L') + df['Engine'].str.findall(r'[0-9].[0-9] L')


# In[19]:


df['Cap'] = df['Cap'].apply(lambda x : '0.0' if len(x) == 0 else ' '.join(x)).str.replace(' ','').str.replace('L','')


# In[20]:


df['Cap'] = df['Cap'].astype(float)


# In[21]:


def test(x):
    if x == 0.0 :
        return np.nan
    else:
        return x
df['Capacity'] = df['Cap'].apply(test).fillna(df['Cap'].mean()).apply(lambda x: round(x,1))


# In[22]:


df.drop('Cap', axis = 1, inplace = True)


# DRIVE TRAIN COLUMN

# In[23]:


df.drop(df[df["Drivetrain"]=='–'].index, inplace=True)


# In[24]:


df['Drivetrain'].unique()


# In[25]:


def drive(x):
    if x == 'Front-wheel Drive' or x == 'Front Wheel Drive' or x == 'FWD':
        return 'FWD'
    if x == 'Four-wheel Drive' or x == 'All-wheel Drive' or x == '4WD' or x == 'AWD':
        return 'AWD'
    if x == 'Rear-wheel Drive' or x == 'RWD':
        return 'RWD'
df['Drivetrain'] = df['Drivetrain'].apply(drive)


# In[26]:


#Removing the rows having data like '-'
df.drop(df[df["FuelType"]=='–'].index, inplace=True)


# In[27]:


def fuel(x):
    if x == 'Gasoline' or x == 'Gasoline Fuel':
        return 'Gasoline'
    if x == 'Electric Fuel System' or x == 'Electric':
        return 'Electric'
    if x == 'Flexible Fuel' or x == 'E85 Flex Fuel':
        return 'Flex'
    else:
        return 'Hybrid'
df['Fuel'] = df['FuelType'].apply(fuel)


# In[28]:


df.drop('FuelType', axis = 1, inplace = True)


# In[29]:


df['State'].unique()


# In[30]:


def state(x):
    if x == 'Michigan' or x == 'US-12':
        return 'CA'
    elif x == 'US-169':
        return 'OK'
    elif x == 'Glens' or x == 'Bldg' or x == 'Suite':
        return 'NY'
    elif x == 'AZ-101':
        return 'AZ'
    else:
        return x


# In[31]:


df['State'] = df['State'].apply(state)


# In[32]:


df['Used/New'].unique()


# In[33]:


#As all cars are certified by its brand so uniquely naming it is Certified
df['Used/New'] = df['Used/New'].apply(lambda x: 'Certified' if 'Certified' in x else x)


# In[34]:


df.drop(['Stock#','VIN', 'Engine', 'StreetName','SellerName', 'ExteriorColor','InteriorColor'], axis = 1, inplace = True)


# In[35]:


df.head(5)


# # DATA VISUALIZATION 

# In[36]:


fig = plt.figure(figsize = (10,10))
sns.barplot(df['Drivetrain'], df['Price'], hue = df['Fuel'])


# For FWD: Hybrid cars have more price followed by Electric, Gasoline and Flex<br/>
# For AWD: Electric cars have more price followed by Hybrid, Gasoline and Flex<br/>
# For RWD: Electric cars have more price followed by Gasoline, Hybrid and Flex<br/>

# In[37]:


fig = plt.figure(figsize = (5,5))
sns.barplot(df['Used/New'], df['Price'])


# Used Cars are less cost than the new cars with certification.

# In[59]:


body = ['Automatic','Manual']
data = df["Transmission"].value_counts()
fig = plt.figure(figsize =(15,5))
colors = ['#DD7596','#8EB897',]
plt.pie(data, labels = body,autopct='%1.2f%%', shadow=True,colors=colors)
plt.title("Most preferred Transmission");


# Most customers prefer to use "Automatic" Gear System Cars.

# In[38]:


fig = plt.figure(figsize = (8,8))
sns.barplot(df['Transmission'], df['Price'], hue = df['Fuel'])


# From the graph it is clear that:
# 
# 1)Price of Electric Automatic > Hybrid Automatic > Gasoline Automatic > Flex Automatic<br/>
# 2)Price of Hybrid Manual > Gasoline Manual.

# In[39]:


fig = plt.figure(figsize = (8,8))
sns.histplot(df['SellerRating'], bins = 20)


# Maximum number of consumer ratings lies in the range of 4 - 5 with 4.7 rating being the most given rating

# In[60]:


fig, ax = plt.subplots(figsize=(15,8))
df["Make"].value_counts().head(10).plot.bar(ax=ax)
plt.title("Top 10 manufacturing company");


# This shows that BMW is the manufacturing company with the largest number. Mercedes-Benz and Toyota complete the top three.

# In[61]:


fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(15,10))
sns.barplot(data=df, x="MinMPG", y="Fuel",palette='YlGn',hue="Drivetrain",ax=ax1)
sns.barplot(data=df, x="MaxMPG", y="Fuel",palette='YlGn',hue="Drivetrain",ax=ax2)
plt.show()


# Here we can see that average fuel consumption is higher in the Electric type.

# In[62]:


sns.countplot(x='Fuel', data=df, orient='h')
plt.title("Most preferred Fuel Type used over the years")


# In[58]:


plt.figure(figsize=(25, 6))
sns.pairplot(df, x_vars=['Year', 'Mileage'], y_vars='Price',size=5, aspect=1.2, kind='scatter')
plt.show()


# # Inference:
# 1) Year of manufacting seems to have a positive correlation with price, which is expected.
# 
# 2) Mileage appears to have a negative correlation with price.

# In[161]:


plt.rcParams['figure.figsize'] = [15,8]
ax = df['Make'].value_counts().plot(kind='bar', colormap = 'Accent')
ax.title.set_text('Numbers of cars sold per company')
plt.xlabel("Car Company",fontweight = 'bold')
plt.ylabel("Count of Cars",fontweight = 'bold')


# # Insights:
# BMW, Mercedes-Benz,Toyota are among the most cars sold as used cars.</br>
# 
# Ferrari, Mercury, and Satum are the lowest sold cars.

# In[48]:


corr=df.corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(corr, cmap='plasma', annot=True, ax=ax,
mask=mask, cbar_kws={'shrink': 0.10},linewidths=.5)
plt.title("Heatmap for Highest correlated features for Cars dataset");


# In[51]:


df.plot(kind='scatter', x='Year', y='Price')


# The scatterplot above shows the relationship between year and price — newer the car is, the more expensive it’s likely to be.

# In[130]:


plt.figure()
sns.pairplot(df, vars=[ 'Year', 'Mileage', 'Price'], hue='Fuel')
plt.show()


# This pairplot gives the observations: those are</BR>
# -- Price varying based on Year and Mileage</br>
# -- AS year increases there is an increase in the Gasoline Engine type.

# In[143]:


#Importing Statistics libraray
from scipy import stats
#Python's statistics is a built-in Python library for descriptive statistics.


# # ANOVA TEST
# ANOVA: Analysis of Variance The Analysis of Variance (ANOVA) is a statistical method used to test whether there are significant differences between the means of two or more groups. ANOVA returns two parameters:
# 
# F-test score: ANOVA assumes the means of all groups are the same, calculates how much the actual means deviate from the assumption, and reports it as the F-test score. A larger score means there is a larger difference between the means.
# 
# P-value: P-value tells how statistically significant our calculated score value is.
# 
# If our price variable is strongly correlated with the variable we are analyzing, we expect ANOVA to return a sizeable F-test score and a small p-value.

# In[147]:


grouped_test1=df[["Transmission","Price"]].groupby(['Transmission'])
grouped_test1.head(2)


# In[149]:


f_val, p_val = stats.f_oneway(grouped_test1.get_group('Automatic')['Price'], grouped_test1.get_group('Manual')['Price'])
print( "ANOVA results: F=%.4f, P-value=%.4f"%(f_val,p_val))  


# In[150]:


grouped_test2=df[["Drivetrain","Price"]].groupby(['Drivetrain'])
grouped_test2.head(2)


# In[151]:


f_val, p_val = stats.f_oneway(grouped_test2.get_group('FWD')['Price'], grouped_test2.get_group('AWD')['Price'],grouped_test2.get_group('RWD')['Price'])
 
print( "ANOVA results: F=%.4f, P-value=%.4f"%(f_val,p_val)) 


# # Conclusion
# 1) With the help of notebook I learnt how exploratory data analysis can be carried out using Pandas plotting.</br>
# 2) Also I have seen making use of packages like matplotlib and seaborn to develop better insights about the data.</br>
# 3) I have seen the impact of columns like mileage, year and Fueltype on the Price i.e.increase/decrease rate.</br>
# 4) The most important inference drawn from all this analysis is, I get to know what are the features on which price is    positively and negatively coorelated with.

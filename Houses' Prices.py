#!/usr/bin/env python
# coding: utf-8

# # Houses'  price prediction project:
# ### I. Problem Definition:
# Goal:
# - Observe the trend of houses' prices over the period
# - Define the characteristics of houses which increased prices
# - See if any relevant between the houses' prices trend in Denvor and the trend of United State housing bubble's period 
# - Predict the sales price for each house
# 
# ### II. Working
# #### 1/ Data preparation
# 1.1 View data
# 
# 1.2 Clean data
# #### 2/ Data analysis
# 2.1 By rooms
# 
# 2.2 By location
# #### 3/ Data modelling
# - Select features
# - Split the dataset to input & target variable. Then split to train, test
# - Train machine learning model
# - Evaluate the model
# 
# 

# In[1]:


# Importing libraries
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
url = "https://raw.githubusercontent.com/pirple/Data-Mining-With-Python/master/Part%202/single_family_home_values.csv"
# Importing the file from the address contained in 'url' into 'df'
df = pd.read_csv(url, sep = ',')
# Showing the first 5 rows of 'df'
df.head(5)


# In[2]:


# Choosing the chart background for matplotlib
plt.style.available
plt.style.use('dark_background')


# In[3]:


# Showing chart background 
plt.plot();


# # 1/ Data preparation

# ## 1.1 View data

# In this part, I do these steps:
# - To see information included (columns' name, the number of rows, data type, number of null and non-null lines...) 
# - To see if there are special points with null lines
# 
#   => Most of null lines are from 'priorSaleDate' and 'priorSaleAmount', these 2 columns have both NaN on same lines. The lines equal 0 in priorSaleAmount are also noticeable because they still have priorSaleDate
# 
#   => 15 houses do not have longtitude and latitude. These houses belong to zipcode 80203-80206, but these areas have many houses. The year they were sold was after 2011, but this period also had many houses sold. Therefore, no significantly common points for these houses. Besides, 15 is a small amount, so I still use latitude and longtitude for further investigation later on
#   
# - In addition, I define the columns which is most relevant with 'priorSaleAmount' in order to fill missing value in this column. 
# 

# In[4]:


# Showing all columns' name and see the total number of rows 
ColumnList = df.columns
IndexList = df.index
display(ColumnList)
display(IndexList)


# In[5]:


# Showing dataframe's information  
df.info()


# In[6]:


# Counting the number of null rows in each column 
df.isnull().sum()


# In[7]:


# Writing function to show null rows in value and percentage
missing_columns = df[['latitude', 'longitude', 'priorSaleDate', 'priorSaleAmount']]
for col in missing_columns:
    missing_data = df[col].isna().sum()
    perc = round((missing_data * 100 / len(df)), 2)
    print(f'{col}: missing entries: {missing_data}, percentage {perc}%')


# In[8]:


# Show missing values of 'priorSaleAmount' to observe the relevant with 'priorSaleDate'
df[df['priorSaleAmount'].isnull()]


# In[9]:


# Show 0 values of 'priorSaleAmount' to observe the relevant with 'priorSaleDate'
df[df['priorSaleAmount'].eq(0)]


# In[10]:


# Show missing values of 'latitude' to observe if there is any common points
df[df['latitude'].isnull()]


# In[11]:


# To see house distribution by zipcode
df['zipcode'].value_counts()


# In[12]:


df.describe()


# In[13]:


# To see the correlation between 'priorSaleAmount' and other elements
df.corr().loc['priorSaleAmount', :].sort_values(ascending=False)


# In[14]:


# To see the correlation between 'estimated_value' and other elements. This is used for data modelling step
df.corr().loc['estimated_value', :].sort_values(ascending=False)


# ## 1.2 Clean data

# For this stage, I modify the data to create new column calculating the price fluctuation per m2:
# - As we can see above, data type of Date column is object. Hence, for calculating purpose, I convert it to date type
# - After calculating the daysdiff column, I find out that there is a line where priorSaleDate after lastSaleDate (min < 0), so I resort 2 columns
# - I fill missing values of 'priorSaleAmount' by median of 'squareFootage', 'bathrooms', 'rooms' respectively following the high of correlation
# 

# In[15]:


# Change type of Date
df['lastSaleDate'] = pd.to_datetime(df.lastSaleDate)
df['priorSaleDate'] = pd.to_datetime(df.priorSaleDate)


# In[16]:


# Calculate days diff to prepare for new variable 
df['daysdiff'] = df['lastSaleDate'] - df['priorSaleDate']
df['daysdiff'] = df['daysdiff'].dt.days
df.head(10)


# In[17]:


# Check if there is any unusual issue
df.daysdiff.describe()


# In[18]:


# Investigate the value which < 0
df[df['daysdiff'] < 0]


# In[19]:


# The data might be put in wrong position, so I resort priorSaleDate and lastSaleDate
df[['priorSaleDate', 'lastSaleDate']] = df.apply(lambda x: pd.Series(sorted([x.priorSaleDate, x.lastSaleDate])), axis = 1)


# In[20]:


# Recal and check days diff again 
df['daysdiff'] = df['lastSaleDate'] - df['priorSaleDate']
df['daysdiff'] = df['daysdiff'].dt.days
df.daysdiff.describe()


# In[78]:


# Creat year diff column
df['diff_Year'] = df['lastSaleDate'] - df['priorSaleDate']
df['diff_Year'] = round(df['diff_Year'] / np.timedelta64(1,'Y'), 2)
df.head()


# In[22]:


# Filling the missing values in priorSaleAmount with the median of squareFootage groups
df['priorSaleAmount'] = df.groupby(['squareFootage'])['priorSaleAmount'].apply(lambda x: x.fillna(x.median()))
df.head(10)


# In[23]:


# Check whether all missing values are filled
pd.set_option('display.max_columns', None)
df[df['priorSaleAmount'].isnull()]


# In[24]:


# Fill the missing values for the 2nd time with the median of bathrooms groups
df['priorSaleAmount'] = df.groupby(['bathrooms'])['priorSaleAmount'].apply(lambda x: x.fillna(x.median()))


# In[25]:


# Check whether all missing values are filled
df[df['priorSaleAmount'].isnull()]


# In[26]:


# Fill the missing values for the 3rd time with the median of rooms groups
df['priorSaleAmount'] = df.groupby(['rooms'])['priorSaleAmount'].apply(lambda x: x.fillna(x.median()))


# In[27]:


# Check whether all missing values are filled
df[df['priorSaleAmount'].isnull()]


# In[28]:


# Calculate new measure which is the fluctuation of price per m2
df['fluct_perm2'] = round(((df.lastSaleAmount - df.priorSaleAmount) / df.squareFootage),2)
df.tail(5)


# In[29]:


# Calculate price per m2 fluctuation each year 
df['fluct_perm2_peryear'] = round(df['fluct_perm2'] / df['diff_Year'], 2)
# Replace infinitive number with 0
df['fluct_perm2_peryear'].replace([np.inf, -np.inf, np.nan], 0, inplace=True)

df.tail(10)


# # 2. Data analysis

# Overall Comments:
# 
# 1/ Houses' prices
# - Over 90% of houses had the price's increase from 0-45$/m2/year. The range for fluctuation was from -8,238-23,182$/m2/year, which was extremely wide
# - From the histogram of price's fluctuation per m2 per year, we can see the shape is leptokurtic, which also means the outliners are significantly fluctuated compared to normal distribution
# 
# 2/ Trend
# - The vibrant period for trading houses was from 1997 to 2017, houses were sold more and more over the years
# - The price increased significantly from 1997 to 2007, from 2007-2008 it dropped dramatically to the bottom. After that it  revised substantially. These movement was quiet in place with the proceedings of bubble housing 2008 
# - Most of the houses' price experienced the increase trend (about 68% of the population). The 2nd place was "no fluctuation" group which was about 27% of total population, and only 5% of houses' price was decreased
# 

# In[32]:


df['yearlastsale'] = pd.DatetimeIndex(df.lastSaleDate).year
df['yearpriorsale'] = pd.DatetimeIndex(df.priorSaleDate).year


# In[33]:


df.groupby('yearlastsale')['lastSaleAmount'].median().plot()
df.groupby('yearpriorsale')['priorSaleAmount'].median().plot()
plt.xlabel('Year')
plt.ylabel('USD')

plt.legend(['lastSaleAmount','priorSaleAmount'])


# In[34]:


df['yearlastsale'].value_counts().plot(kind = 'bar')


# In[35]:


# See some information of new variable
df['fluct_perm2_peryear'].describe()


# In[36]:


# Draw boxplot to see if there are many outliners
sns.boxplot(y=df["fluct_perm2_peryear"] );
#plt.ylim(-10, 100)
plt.show()


# In[37]:


# Draw histogram to see the central tendency of data
plt.figure(figsize=(15,10))
meadian =  90.190341
color = '#fc4f30'
plt.axvline(meadian, color=color, label='Price fluct meadian') 
plt.hist(df['fluct_perm2_peryear'], bins=500, log=True);


# In[38]:


# Closer look into histogram 
plt.figure(figsize=(15,10))
bins = [-1000, -700, -500, -400, -300, -200, -100, -50, 0, 50, 100, 200, 300, 400, 500, 700, 1000]
meadian =  90.190341
color = '#fc4f30'
plt.axvline(meadian, color=color, label='Price fluct meadian') 
plt.hist(df['fluct_perm2_peryear'], bins=bins, edgecolor='white', log=True);


# In[39]:


# Grouping the fluctuation of price per m2 per year
df['Trend'] = df['fluct_perm2_peryear'].apply(lambda x: 'increase' if x>0 else ('decrease' if x<0 else 'no fluctuation'))
df['Increase'] = df['Trend'].apply(lambda y: 1 if y == 'increase' else 0)
df['Decrease'] = df['Trend'].apply(lambda z: 1 if z == 'decrease' else 0)
df['No_fluct'] = df['Trend'].apply(lambda w: 1 if w == 'no fluctuation' else 0)
df.head(10)


# In[61]:


# Calculate % of each group
trend = df.groupby(['Trend'])['Trend']
counts = trend.count()
percent100 = round((counts * 100 / len(df['Trend'])), 2).astype(str) + '%'
pd.DataFrame({'counts': counts, 'percent': percent100})


# ## 2.1 By rooms

# Comments:
# 
# - Over 90% of houses had 2-3 bedrooms and 1-3 bathrooms
# - Houses which have 2 bedrooms and 2 bathrooms accounted for the highest price's fluctuation (both increase and decrease). Those houses which have 1, 3 or 4 bedrooms also increased much. These kinds of houses seemed to fit with single person, couples or small families
# 
# - The house which had the highest number of room was decreased in price
# 
# Hence, it seems like the more rooms the houses had, the lower prices they were

# In[62]:


sns.boxplot(y=df["bathrooms"] );

#plt.ylim(-20, 100)
plt.show()


# In[63]:


# plt.figure(figsize=(15,10))
plt.hist(df['bathrooms'], bins=50);


# In[64]:


# plt.figure(figsize=(14,10))
br, fmy = df['bathrooms'], df['fluct_perm2_peryear']
plt.bar(x=br, height=fmy)
plt.xticks(br)
plt.xlabel('Bathrooms')
plt.ylabel('Fluctuation per m2 per year')
plt.show


# In[79]:


df1 = df[df['bathrooms'].isin([6])]
df1.groupby(['zipcode'])['bathrooms'].count()


# In[65]:


sns.boxplot(y=df["bedrooms"] );

# plt.ylim(0, 5)
plt.show()


# In[66]:


# plt.figure(figsize=(15,10))
plt.hist(df['bedrooms'], bins=50);


# In[67]:


# plt.figure(figsize=(10,8))
bedr, fmy1 = df['bedrooms'], df['fluct_perm2_peryear']
plt.bar(x=bedr, height=fmy1)
plt.xticks(bedr)
plt.xlabel('Bedrooms')
plt.ylabel('Fluctuation per m2 per year')
plt.show


# In[68]:


plt.figure(figsize=(15,8))
r, fmy2 = df['rooms'], df['fluct_perm2_peryear']
plt.bar(x=r, height=fmy1)
plt.xticks(r)
plt.xlabel('Rooms')
plt.ylabel('Fluctuation per m2 per year')
plt.show


# ## 2.2 By location

# Comments:
# 
# 1/ Analyze by location
# 
# - As we can see from the chart below, most of the citizens live in the central of the city. That's why most of the increase appeared in that area
# 
# - 2 highest price increase located in Villa Park (west of the central area), and Cherry Creek (east of the city)
# 
#  +Villa Park, quiet area with wonderful nature (river, mountain, hill, park, lake...) fit for rich businessman to relax. Nearby area also has bar, pub to entertain 
#  +Cheery Creek: has 2 big commercial centers, a blend of locally-owned businesses and nationally recognized brands, including a variety of shops, restaurants, salons, spas and service providers. Fillmore Plaza, located on Fillmore Street between First and Second Avenues, is home to numerous community events including Cherry Creek North Food & Wine in summer and the Cherry Creek Arts Festival
# https://i.pinimg.com/originals/f0/6a/1f/f06a1f8bc800da0e0c4ad6d8d89de0dd.jpg
# 
# 2/ The relevant with US housing bubble
# 
# - The density of the increase trend grew from 1997 to 2008 and drop afterwards for about 7-8 years before it recovered again in 2016
# 
# -> This trend is quite in line with the house market situation in USA. The unprecedented increase in house prices starting in 1997, the housing bubble in 2008 made the whole market collapsed for a decade until the recovery from 2016
# 
# - We can see most of the decrease also appeared in this period from 2008-2015
# - However, because Colorado was not the state which experienced huge impact from the crisis, the main population trend was still increase
# https://en.wikipedia.org/wiki/United_States_housing_bubble

# In[80]:


df.groupby('zipcode')['fluct_perm2_peryear'].mean().plot(kind = 'bar')


# In[81]:


df.corr().loc['fluct_perm2_peryear', :].sort_values(ascending=False)


# In[99]:


### Show the location of zipcode on map
df['zipcode'] = df['zipcode'].astype(str)
plt.figure(figsize=(15,8))
plt.style.use('seaborn')
sns.scatterplot(data=df, x="longitude", y="latitude", hue="zipcode", palette="tab10")


# In[112]:


### Fluctuation trend by longitude & latitude
plt.figure(figsize=(15,8))
plt.style.use('seaborn')
sns.scatterplot(data=df, x="longitude", y="latitude", hue="Trend", palette="icefire")


# In[109]:


### Fluctuation price per m2 per year by longitude & latitude

lat, lon = df['latitude'], df['longitude']
fluctpm2py, zc = df['fluct_perm2_peryear'], df['zipcode']

plt.style.use('seaborn')
plt.figure(figsize=(16,10))

# Plot using Pyplot API
plt.scatter(lon, lat, 
            c=fluctpm2py, cmap='twilight_shifted',
            s=fluctpm2py, linewidths=0, alpha=0.5)

# plt.axis('equal')
plt.xlabel('longlatitude')
plt.ylabel('latitude');
plt.colorbar(label='Fluctuation per m2 per year')
# plt.clim(100, 20000)


# # 3. Modeling (Regression)

# ### Regression

# ### Comment:
# 
# I ran model to predict the houses' price. The variance compared to expert's prediction is 175934

# In[114]:


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm


# In[115]:


# Select features
X = df[['bedrooms', 'bathrooms', 'rooms','squareFootage','lotSize', 'priorSaleAmount', 'lastSaleAmount']]
y = df['estimated_value']


# In[116]:


X.info()


# In[117]:


X.head()


# In[118]:


#Split the dataset to input & target variable. Then split to train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)


# In[119]:


X_train.shape, y_train.shape


# In[120]:


X_test.shape, y_test.shape


# In[121]:


#Train machine learning model
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state = 1)
rf_model.fit(X_train, y_train)


# In[122]:


rf_predict = rf_model.predict(X_test)


# In[123]:


pd.DataFrame({'y': y_test, 'predict': rf_predict})


# In[124]:


#Evaluate the model
mse = mean_squared_error(y_test, rf_predict)
mse


# In[125]:


import math
rmse = math.sqrt(mse)
print(rmse)


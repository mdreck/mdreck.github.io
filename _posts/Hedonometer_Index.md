# The Hedonometer Index: A Happiness Forecast 

By: Matthew Reck

*This Jupyter notebook was presented to an audience of professors, peers, and professionals from various backgrounds as part of a Graduate Capstone project.*
___

The following analysis was performed on the Hedonometer Index from the University of Vermont (http://hedonometer.org/index.html). 

The Hedonometer Index measures the daily sentiment or attitude of Twitter. It is a database of 10,000 unique words that are individually scored on a nine-point scale of happiness: 1 (sad/bad) to 9 (happy/good). Words associated with good or happiness such as love and laughter have very high scores, whereas words like death and destruction have very low scores. The index analyzes Twitter's Gardenhose feed, a random sample of 10% of daily tweets, and calculates the average score for the words used on that particular day. The Hedonometer Index has measured the average happiness score on a daily basis since September of 2008.

This analysis will walk through a brief exploratory data analysis (EDA) to determine the dates with the highest and lowest scores, and seeks to identify interesting trends and seasonalities within the Hedonometer Index. The "holiday effect", for example, is the tendancy for stock prices to increase on the last trading day before a national holiday. We will investigate what influence, if any, the holiday effect has on the Hedonometer Index. Lastly, we will demonstrate the power and intuitiveness of Facebook's Prophet library to forecast Twitter's future happiness.

The purpose of this analysis is not to serve as an exhaustive approach to time series forecasting, but rather to provide readers with a strong and straightforward introduction into EDA and Facebook's Prophet library. Let's get started!

### Import libraries


```python
# Import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Use seaborn style defaults
sns.set()

# Info
__author__ = "Matthew Reck"
__email__ = "mdreck@gmail.com"
__linkedin__ = "linkedin.com/in/mdreck/"
```

### Load data


```python
# Load data and set date as index
raw_df = pd.read_csv('C:/Users/garcr/data/sumhapps.csv',
                     parse_dates=['date'], index_col='date')

# View first five rows of dataframe
print(raw_df.head())

# View datetime index
print('\n', raw_df.index)
```

                   value
    date                
    2008-09-09  6.042504
    2008-09-10  6.027954
    2008-09-11  6.020410
    2008-09-12  6.027490
    2008-09-13  6.035040
    
     DatetimeIndex(['2008-09-09', '2008-09-10', '2008-09-11', '2008-09-12',
                   '2008-09-13', '2008-09-14', '2008-09-15', '2008-09-16',
                   '2008-09-17', '2008-09-18',
                   ...
                   '2020-02-02', '2020-02-03', '2020-02-04', '2020-02-05',
                   '2020-02-06', '2020-02-07', '2020-02-08', '2020-02-09',
                   '2020-02-10', '2020-02-11'],
                  dtype='datetime64[ns]', name='date', length=4150, freq=None)
    

### Data Exploration

Before moving into the modeling phase, we should first investigate the structure, distribution, and characteristics of the data. 

#### Examine the data


```python
def initial_analysis(df):
    """
    Given a dataframe and returns a simple report of:
        - Shape of dataframe 
        - Columns and data types
        - Null values and duplicate values
    """
    print('Report of Initial Data Analysis:\n')
    print(f'Shape of dataframe: {raw_df.shape}')
    print(f'\n Features and Data Types: {raw_df.dtypes}')
    print(f'\n Null values: {raw_df.isnull().sum()}')
    print(f'\n Duplicate dates: {raw_df.index.duplicated().sum()}')


initial_analysis(raw_df)
```

    Report of Initial Data Analysis:
    
    Shape of dataframe: (4150, 1)
    
     Features and Data Types: value    float64
    dtype: object
    
     Null values: value    0
    dtype: int64
    
     Duplicate dates: 0
    

- There are no missing values or duplicate dates present in the dataset

##### Rename columns


```python
# Change column name 'value' to 'Score'
df = raw_df.copy()
df.rename(columns={'value': 'Score'}, inplace=True)

# Print new column names
print('New column names:\n', df.columns)
```

    New column names:
     Index(['Score'], dtype='object')
    

##### Visualize and test data for normality 


```python
# Visualize dataset using a boxplot and a histogram with a density plot
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(df.Score)
plt.subplot(1, 2, 2)
sns.distplot(df.Score, bins=20)
plt.show()
```


![png](Hedonometer_Index_files/Hedonometer_Index_14_0.png)



```python
# Determine the normality and shape of the data
from scipy.stats import norm, kurtosis
from scipy.stats import skew
skew = skew(df.Score)
kurtosis = kurtosis(df.Score)

print('Skewness: ', '%.2f' % skew)
print('Kurtosis: ', '%.2f' % kurtosis)
```

    Skewness:  0.52
    Kurtosis:  2.80
    

- The boxplot identified a number of outliers within the dataset, particularly those greater than the maximum limit, and further supported by the positive skewness (0.52). The dataset is slightly skewed to the right by large values.  
- The long tails of the density plot and the data's kurtosis (2.8) indicate that the dataset has a high percentage of outliers
- Further investigation of the outliers is necessary  

#### Identify and inspect potential outliers 


```python
# Use the interquartile range (IQR) to identify outliers
stat = df.Score.describe()
print(stat)
IQR = stat['75%'] - stat['25%']
upper = stat['75%'] + 1.5 * IQR
lower = stat['25%'] - 1.5 * IQR
print('\nThe upper and lower limits for potential outliers are {} and {}'.format(
    '%.3f' % upper, '%.3f' % lower))
```

    count    4150.000000
    mean        6.019237
    std         0.050957
    min         5.744207
    25%         5.980616
    50%         6.015866
    75%         6.056489
    max         6.375729
    Name: Score, dtype: float64
    
    The upper and lower limits for potential outliers are 6.170 and 5.867
    


```python
# Inspect the outliers above the upper limit ( > 6.170 )
df[df.Score > upper].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2008-11-27</td>
      <td>6.193093</td>
    </tr>
    <tr>
      <td>2008-12-24</td>
      <td>6.254525</td>
    </tr>
    <tr>
      <td>2008-12-25</td>
      <td>6.375729</td>
    </tr>
    <tr>
      <td>2009-01-01</td>
      <td>6.192567</td>
    </tr>
    <tr>
      <td>2009-04-12</td>
      <td>6.187767</td>
    </tr>
    <tr>
      <td>2009-11-26</td>
      <td>6.241893</td>
    </tr>
    <tr>
      <td>2009-12-24</td>
      <td>6.257590</td>
    </tr>
    <tr>
      <td>2009-12-25</td>
      <td>6.342121</td>
    </tr>
    <tr>
      <td>2009-12-31</td>
      <td>6.173756</td>
    </tr>
    <tr>
      <td>2010-01-01</td>
      <td>6.178530</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Inspect the outliers below the lower limit ( < 5.867 )
df[df.Score < lower].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2016-06-12</td>
      <td>5.825362</td>
    </tr>
    <tr>
      <td>2016-07-08</td>
      <td>5.859895</td>
    </tr>
    <tr>
      <td>2017-08-12</td>
      <td>5.851742</td>
    </tr>
    <tr>
      <td>2017-10-02</td>
      <td>5.744207</td>
    </tr>
    <tr>
      <td>2017-10-03</td>
      <td>5.860612</td>
    </tr>
    <tr>
      <td>2018-02-15</td>
      <td>5.850913</td>
    </tr>
    <tr>
      <td>2019-03-15</td>
      <td>5.832015</td>
    </tr>
    <tr>
      <td>2019-08-04</td>
      <td>5.757763</td>
    </tr>
    <tr>
      <td>2019-08-05</td>
      <td>5.831435</td>
    </tr>
    <tr>
      <td>2020-01-03</td>
      <td>5.781341</td>
    </tr>
  </tbody>
</table>
</div>



All but one of the 34 upper limit outliers occur on the following holidays: Christmas Eve, Christmas Day, Thanksgiving, New Year's Eve, New Years Day, Valentine's Day, and Easter (4/12/2009). The lone exception is May 9th, 2010. Due to the recurring nature of the holidays, these dates will routinely produce happiness scores greater than the upper limit and should be considered natural variation. 

The 10 outliers falling below the lower limit are all associated with mass shootings, hate crimes, acts of terrorism, and most recently the fear resulting from the successful airstrike on Iran's General Soleimani (Las Vegas, Orlando, El Paso, Dayton, Parkland, Charlottesville, Boston Marathon, Dallas Police). While these events may appear to be random, they capture valuable information and will be included in the analysis.    

The outliers at both ends of the spectrum appear to be legitimate. There is no reason to believe that they are the result of missing or corrupt data and we will not remove them from the dataset. 

#### Time series visualization

Visualize the time series to identify potential seasonality and trends


```python
# Plot happiness index time series
df.plot(y='Score', title='Hedonometer Index', figsize=(15, 10))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x25938605f88>




![png](Hedonometer_Index_files/Hedonometer_Index_23_1.png)


The yearly seasonality resulting from the holiday season is evident with the top 20 scores occuring on either Christmas Day, Christmas Eve, or Thanksgiving. 

#happiness_df.sort_values('Score', ascending=False).head(20)

Conversely, the emergence of violence since mid 2016 is exceedingly apparent. There are several trends (instances of increasing or decreasing slopes) within the time series. Further investigation is required to determine the current trend, and any overarching or cyclical behavior within the Hedonometer Index.

#### Visualize the average scores by day, month, and year

Calculate average scores and explore the subsets for insight 

##### Aggregate data (Groupby)


```python
# Add columns with year, month, and weekday to a copy of the df
happiness_df = df.copy()
happiness_df['Year'] = happiness_df.index.year
happiness_df['Month'] = happiness_df.index.month
happiness_df['Weekday'] = happiness_df.index.weekday_name

# Verify changes
happiness_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score</th>
      <th>Year</th>
      <th>Month</th>
      <th>Weekday</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2008-09-09</td>
      <td>6.042504</td>
      <td>2008</td>
      <td>9</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <td>2008-09-10</td>
      <td>6.027954</td>
      <td>2008</td>
      <td>9</td>
      <td>Wednesday</td>
    </tr>
    <tr>
      <td>2008-09-11</td>
      <td>6.020410</td>
      <td>2008</td>
      <td>9</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <td>2008-09-12</td>
      <td>6.027490</td>
      <td>2008</td>
      <td>9</td>
      <td>Friday</td>
    </tr>
    <tr>
      <td>2008-09-13</td>
      <td>6.035040</td>
      <td>2008</td>
      <td>9</td>
      <td>Saturday</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create groupby function to average weekday, monthly, and yearly scores
def avg_score(col):
    '''return average score by column'''
    return happiness_df.groupby(col).agg({'Score': 'mean'})


# Calculate average scores
monthly_score = avg_score('Month')
yearly_score = avg_score('Year')
daily_score = avg_score('Weekday')

print(daily_score)

# Reindex daily_score to order Weekdays from Monday to Sunday
workweek = ['Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_score = daily_score.reindex(workweek)

print('\nOrdered daily_score dataframe: \n', daily_score)
```

                  Score
    Weekday            
    Friday     6.032062
    Monday     6.010960
    Saturday   6.032461
    Sunday     6.023219
    Thursday   6.016499
    Tuesday    6.009139
    Wednesday  6.010445
    
    Ordered daily_score dataframe: 
                   Score
    Weekday            
    Monday     6.010960
    Tuesday    6.009139
    Wednesday  6.010445
    Thursday   6.016499
    Friday     6.032062
    Saturday   6.032461
    Sunday     6.023219
    

##### Bar charts
__Weekday__


```python
# Plot the average score for each day of the week
plt.figure(figsize=(7, 5))
plt.bar(daily_score.index, daily_score.Score)
plt.ylim(6.005, 6.035)
plt.xticks(rotation=45)
plt.title('Average Weekday Score')
plt.show()
```


![png](Hedonometer_Index_files/Hedonometer_Index_30_0.png)


The bar chart displays the average score for each day of the week. Perhaps unsurprisingly, we observe that happiness tends to bottom out at the start of the work week and the scores steadily increase into the weekend (weekend-bump).

__Monthly__


```python
# Plot the average score for each month
plt.figure(figsize=(7, 5))
plt.bar(monthly_score.index, monthly_score.Score)
plt.ylim(6.00, 6.05)
plt.xticks(monthly_score.index)
plt.title('Average Monthly Score')
plt.show()
```


![png](Hedonometer_Index_files/Hedonometer_Index_32_0.png)


The average monthly score demonstrates the holiday effect and seasonality witnessed in the time series visualization with November and December having higher average scores than the others. There is a notable drop in monthly scores between May and October.     

__Yearly__


```python
# Plot the average score for each year
plt.figure(figsize=(7, 5))
plt.bar(yearly_score.index, yearly_score.Score)
plt.ylim(5.90, 6.08)
plt.xticks(yearly_score.index, rotation=90)
plt.title('Average Yearly Score')
plt.show()
```


![png](Hedonometer_Index_files/Hedonometer_Index_34_0.png)


- The yearly average indicates a strong downward trend since 2015
- 2020 is on track to be the unhappiest year to date
- If happiness is cyclical, scores should increase in the upcoming years

##### Boxplots


```python
# Create year-wise, month-wise,
fig, axes = plt.subplots(3, 1, figsize=(15, 12))
for name, ax in zip(['Year', 'Month', 'Weekday'], axes):
    sns.boxplot(data=happiness_df, x=name, y='Score', ax=ax)
    ax.set_ylabel('Score')
```


![png](Hedonometer_Index_files/Hedonometer_Index_37_0.png)


There appears to be the presence of an oscillatory pattern; however, because the dataset only has 11 full years of data ('09 - '19) it is premature to consider it cyclical. The yearly boxplot seems to indicate that the rise of violence and increased political tension appears to have negatively impacted the yearly scores. This is illustrated by the large number of lower limit outliers ('16-'19) and the current declining trend. 

In observing the monthly breakdown, December has an extreme number of outliers attributed to the holiday effect. Due to these outliers, November and December have relatively dispersed data, while March and September have less variation. 

The upper limit weekday outliers primarily consist of the holidays we explored earlier.  

### Examine correlation (happiness & time horizon)

Explore the relationship between the independent variables (time horizons) and the dependent variable (score). The time horizons are yearly, monthly, and for each day of the week.  

#### Create dummy variables
Weekdays must be converted to numerical "dummy" variables to examine correlation


```python
def one_hot_encode_feature_df(df, cat_vars=None, num_vars=None):
    '''
    Performs one-hot encoding on all categorical variables
    Combines result with continous variables
    '''
    cat_df = pd.get_dummies(df[cat_vars])
    num_df = df[num_vars].apply(pd.to_numeric)
    return pd.concat([cat_df, num_df], axis=1)


# Define Categorical vars for dummy transformation, and numerical vars to merge
categorical_vars = ['Weekday']
numerical_vars = ['Month', 'Year']

# Encode categorical data and add Score to cat_df
cat_df = one_hot_encode_feature_df(
    happiness_df, cat_vars=categorical_vars, num_vars=numerical_vars)
cat_df['Score'] = happiness_df.Score
cat_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Weekday_Friday</th>
      <th>Weekday_Monday</th>
      <th>Weekday_Saturday</th>
      <th>Weekday_Sunday</th>
      <th>Weekday_Thursday</th>
      <th>Weekday_Tuesday</th>
      <th>Weekday_Wednesday</th>
      <th>Month</th>
      <th>Year</th>
      <th>Score</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2008-09-09</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>6.042504</td>
    </tr>
    <tr>
      <td>2008-09-10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>2008</td>
      <td>6.027954</td>
    </tr>
    <tr>
      <td>2008-09-11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>6.020410</td>
    </tr>
    <tr>
      <td>2008-09-12</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>6.027490</td>
    </tr>
    <tr>
      <td>2008-09-13</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>6.035040</td>
    </tr>
  </tbody>
</table>
</div>



#### Visualize correlation (heatmap)


```python
def heatmap_numeric_w_dependent_variable(df, dependent_variable):
    ''' 
    Returns heatmap of the correlation between independent and dependent variable(s) 
    '''
    plt.figure(figsize=(4, 5))
    g = sns.heatmap(df.corr()[[dependent_variable]].sort_values(by=dependent_variable, ascending=False),
                    annot=True,
                    cmap='coolwarm',
                    vmin=-1,
                    vmax=1)
    return g


heatmap_numeric_w_dependent_variable(cat_df, 'Score')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x259385e0a88>




![png](Hedonometer_Index_files/Hedonometer_Index_44_1.png)


According to the heatmap, Friday and Saturday have the strongest positive relationship with happiness as we previously discussed. Month is also positively related to happiness, so on average, as the month integer increases (1 --> 12) so do the happiness scores. That result is largely attributed to the high scores of November and December (holiday effect). The relationship between year and happiness is negative, implying that happiness is declining into the future.     

#### Plot the various time horizons 


```python
# Resample to weekly, monthly, and yearly frequencies, aggregate with mean
weekly_mean = happiness_df.Score.resample('W').mean()
monthly_mean = happiness_df.Score.resample('M').mean()
yearly_mean = happiness_df.Score.resample('Y').mean()

# Plot the resampled time series together
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(happiness_df.Score, label='Daily')
ax.plot(weekly_mean, label='Weekly Mean Resample')
ax.plot(monthly_mean, label='Monthly Mean Resample')
ax.plot(yearly_mean, label='Yearly Mean Resample')
ax.set_ylabel('Score')
ax.legend()
```




    <matplotlib.legend.Legend at 0x25938640d48>




![png](Hedonometer_Index_files/Hedonometer_Index_47_1.png)


The daily scores were resampled to a lower frequency (downsampling) using the weekly, monthly, and yearly means. Downsampling removes some of the variability from the daily time series through aggregation, enabling us to analyze the data on various time scales. 

While not as extreme, the seasonality is still present in the weekly and monthly aggregates. The yearly mean resample illustrates the yearly trends within the Hedonometer Index. Through the first month and a half of 2020 it is continuing the downward yearly trend.

### Decompose for trend and seasonality

Further investigate the seasonalities and trends of the index through seasonal decomposition


```python
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams

# Additive Decomposition
result_add = seasonal_decompose(
    df['Score'], model='additive', extrapolate_trend='freq', freq=365)

# Plot
plt.rcParams.update({'figure.figsize': (10, 10)})
result_add.plot().suptitle('Additive Decomposition', fontsize=22)
plt.show()
```


![png](Hedonometer_Index_files/Hedonometer_Index_51_0.png)


Additive decomposition was selected as the magnitude of the seasonal fluctuations did not vary in proportion to the level of the time series.

The trend and seasonality plots confirm our findings from the exploratory data analysis. The current negative trend is particularly explicit and doesn't appear to be slowing down. To find out, we will employ Facebook's Prophet to forecast the future values of the time series.       

### Forecasting with Prophet

Facebook's Prophet, is a robust additive regression model that provides intuitive parameters to handle yearly, weekly, and daily seasonality, as well as holiday effects. It is most effective when working with time series data that contains strong seasonal effects and multiple seasons of historical data.

Prophet essentially automates much of the analysis we've performed to this point, in addition to forecasting future values in a few simple, but powerful, lines of code. We'll plot the forecast, break it down via decomposition, and examine its performance.  


```python
from fbprophet import Prophet

# Prophet requires columns be named ds (Date) and y (value)
# 'ds' requires a date or datetime object, 'y' must be numeric
df1 = raw_df.copy()
df1.reset_index(level=0, inplace=True)
df1 = df1.rename(columns={'date': 'ds', 'value': 'y'})

# Initialize prophet model and fit the data
# Increase confidence interval to 95% (Prophet default is 80%)
prophet = Prophet(yearly_seasonality=True, interval_width=0.95)
# Include the holiday effects in the model
prophet.add_country_holidays(country_name='US')
prophet.fit(df1)
```

    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    




    <fbprophet.forecaster.Prophet at 0x25938ea8e88>



*(Prophet disabled daily seasonality because our data is aggregated at the daily level and is not sub-daily or hourly, which Prophet requires to determine daily seasonality)*


```python
# Make a future dataframe for 2 years
future = prophet.make_future_dataframe(periods=365 * 2, freq='D')
# Make predictions
forecast = prophet.predict(future)
# Looking at the last 5 days that the model is predicting for two years out
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>yhat</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>4875</td>
      <td>2022-02-06</td>
      <td>5.930244</td>
      <td>5.711248</td>
      <td>6.144044</td>
    </tr>
    <tr>
      <td>4876</td>
      <td>2022-02-07</td>
      <td>5.917782</td>
      <td>5.704303</td>
      <td>6.128697</td>
    </tr>
    <tr>
      <td>4877</td>
      <td>2022-02-08</td>
      <td>5.916652</td>
      <td>5.703404</td>
      <td>6.140386</td>
    </tr>
    <tr>
      <td>4878</td>
      <td>2022-02-09</td>
      <td>5.917918</td>
      <td>5.705776</td>
      <td>6.141114</td>
    </tr>
    <tr>
      <td>4879</td>
      <td>2022-02-10</td>
      <td>5.921007</td>
      <td>5.716079</td>
      <td>6.129652</td>
    </tr>
  </tbody>
</table>
</div>



If we look at the bottom row of the table above:

- On February 10th, 2020 Prophet forecasts that the happiness score will be ~ 5.92 (yhat)
- The 95% confidence interval puts the value between ~ 5.69 and 6.14 
  - Prophet is 95% confident that on 2/10/2022 the score will fall between those two values

Let's take a closer look at the forecast and calculate some performance metrics.

#### Visualizing the forecast 


```python
# Plot the historical fitted values and the two year forecast
prophet.plot(forecast, xlabel='Date', ylabel='Score')
plt.title('Hedonometer Index')
```




    Text(0.5, 1, 'Hedonometer Index')




![png](Hedonometer_Index_files/Hedonometer_Index_60_1.png)


- black dots represent the actual measurements
- the blue line is the forecast
- light blue window is the 95% confidence bound

Prophet predicts the trend of declining sentiment will progress into the future. We see the model's uncertainty (upper and lower confidence bounds) grows as the calendar turns over into 2021 and beyond.     

#### Plot the components

Recall that Prophet is an additive model and its forecast is composed of a non-seasonal trend and various seasonalities (yearly, weekly, and holidays in our case). The forecast can be decomposed in a similar fashion to the additive decomposition we performed above using only the historical data. Prophet performs its own weekday dummy variable transformation in determining the weekly seasonality.

As an additive model, the Y axis values are the incremental effect on the dependent variable (happiness) by the seasonal component or independent variable (holidays, weekly, yearly). The trend plot is separate from the seasonalities in this regard. We will further examine the incremental effects below.  


```python
# Plot the trends and patterns
fig = prophet.plot_components(forecast)
```


![png](Hedonometer_Index_files/Hedonometer_Index_64_0.png)


trend: starting in mid-2015 there is a discernible downward trend in the Hedonometer Index's happiness score. Prophet forecasts that the average score for 2022 will drop to approximately 5.9.  

holidays: this chart displays the holiday effect and seasonality that we've been discussing. In general, holidays have an overwhelmingly positive impact on the score for that particular day. For example, Christmas day increases happiness by ~0.24 points. Whereas Martin Luther King day has a small negative effect on the score, presumably due to the negative connotation associated with words like death and assassination.      

weekly: the weekly seasonality supports the weekend-bump hypothesis with scores peaking over the weekend. Friday, Saturday, and Sunday had positive effects on happiness, adding ~0.013, 0.015, and 0.005 points respectively. While the remaining weekdays had a negative influence (M, T, W, Th).        

yearly: the yearly pattern is interesting as scores steadily decline throughout the summer months with a sharp increase at year end, supporting the holiday effect. In mid-January we can see the post-holiday blues kicking in and negatively affecting happiness by ~0.015 points.  

To summarize, these components indicate that longterm happiness should maintain its downward trend. However, if scores can stabilize throughout the summer months, the likelihood of a year over year increase is strong due to the positive influence of the holiday effect in November and December.   

#### Performance and diagnostics

After visualizing the forecast, trend, and seasonalities, one question still remains: "how good is the model?" Thankfully, Prophet maintains a number of diagnostic tools to examine the model's performance. It uses cross-validation on historical data to assess the accuracy of the forecast. By selecting cutoff points in the historical data and then fitting the model up to those points, we can compare the forecasted values to the actual observations. 

Prophet requires the forecast horizon (horizon), and optionally the size of the initial training period (initial) and the length of time between cutoff dates (period). By default, it sets the initial training period to three times the given horizon, and sets the cutoff to half of the horizon. The initial training period must be long enough to capture the components and seasonality of the model, therefore requiring at least a year to detect yearly seasonality.

We're going to use similar values to Prophet's default parameters, and see how the model performs in forecasting 365 days into the future. Beginning with 3 years of training data (1095 days) and making predictions every 180 days (16 forecasts).      


```python
from fbprophet.diagnostics import cross_validation

# set forecast horizon to 365 days
# set initial training period of 1095 days (365 * 3)
# set cutoff period of 180 days (~365 * .5)
cv_results = cross_validation(
    prophet, horizon='365 days', initial='1095 days', period='180 days')
cv_results.head()
```

    INFO:fbprophet:Making 16 forecasts with cutoffs between 2011-09-21 00:00:00 and 2019-02-11 00:00:00
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>yhat</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>y</th>
      <th>cutoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2011-09-22</td>
      <td>5.968906</td>
      <td>5.932937</td>
      <td>6.002964</td>
      <td>5.975924</td>
      <td>2011-09-21</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2011-09-23</td>
      <td>5.990840</td>
      <td>5.955160</td>
      <td>6.023744</td>
      <td>6.014901</td>
      <td>2011-09-21</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2011-09-24</td>
      <td>5.990706</td>
      <td>5.951098</td>
      <td>6.027809</td>
      <td>6.003657</td>
      <td>2011-09-21</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2011-09-25</td>
      <td>5.983623</td>
      <td>5.946236</td>
      <td>6.018709</td>
      <td>5.973239</td>
      <td>2011-09-21</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2011-09-26</td>
      <td>5.962995</td>
      <td>5.927894</td>
      <td>5.998942</td>
      <td>5.970459</td>
      <td>2011-09-21</td>
    </tr>
  </tbody>
</table>
</div>



The table above displays the actual score (y) observed on the date (ds) and the cutoff date used to make the prediction (yhat). Therefore, the first row states that on 9-22-2011, one day after the cutoff, the model predicted a price of ~5.969 with an observed value of ~5.976. The forecast was under by 0.007 or 0.12% (percent error). This is just one example of a residual-based metric and Prophet provides its own set of performance metrics we will look into.  


```python
from fbprophet.diagnostics import performance_metrics

# display Prophet's performance metrics
prophet_perf = performance_metrics(cv_results)
prophet_perf.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>horizon</th>
      <th>mse</th>
      <th>rmse</th>
      <th>mae</th>
      <th>mape</th>
      <th>coverage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>324</td>
      <td>361 days</td>
      <td>0.002848</td>
      <td>0.053366</td>
      <td>0.041770</td>
      <td>0.006967</td>
      <td>0.956897</td>
    </tr>
    <tr>
      <td>325</td>
      <td>362 days</td>
      <td>0.002824</td>
      <td>0.053138</td>
      <td>0.041829</td>
      <td>0.006976</td>
      <td>0.957759</td>
    </tr>
    <tr>
      <td>326</td>
      <td>363 days</td>
      <td>0.002805</td>
      <td>0.052958</td>
      <td>0.041855</td>
      <td>0.006979</td>
      <td>0.956897</td>
    </tr>
    <tr>
      <td>327</td>
      <td>364 days</td>
      <td>0.002810</td>
      <td>0.053012</td>
      <td>0.041840</td>
      <td>0.006976</td>
      <td>0.955172</td>
    </tr>
    <tr>
      <td>328</td>
      <td>365 days</td>
      <td>0.002820</td>
      <td>0.053104</td>
      <td>0.041757</td>
      <td>0.006962</td>
      <td>0.955172</td>
    </tr>
  </tbody>
</table>
</div>



The last row reads that when forecasting with a horizon of 365 days, the mean absolute percent error (MAPE) is ~0.007 or 0.7%. While that is a strong prediction, the entire range of the historical dataset is only ~0.63 (6.37 - 5.74) or about a 10% difference. Upon further investigation into the volatility of the dataset, we found that the largest difference between consecutive days was less than 4%.
#volatility = np.log(df['Score'] / df['Score'].shift()) ; plt.plot(volatility)

According to the Prophet's documentation, Facebook prefers the mean absolute percentage error in determining the accuracy of a forecast, and we'll follow their recommendation in our analysis. Effectively out of the box, and with very little parameter tuning, Prophet was able to achieve a 0.696% MAPE on the 365th day of the forecast. We can visualize the performance of the model (MAPE) throughout the 365 day forecast.     


```python
from fbprophet.plot import plot_cross_validation_metric

# Plot MAPE to visualize Prophet performance
fig = plot_cross_validation_metric(cv_results, metric='mape')
```


![png](Hedonometer_Index_files/Hedonometer_Index_72_0.png)


The blue line represents the MAPE, and the dots represent the absolute percent error for each prediction. From the chart, we can determine that errors are typically around .3% for predictions that are around a month into the future, and steadily increase to roughly .7% for predictions in a year from now. The greatest percent error for any single observation was approaching 5% (the dot in the upper right at ~280 days).

To get a better understanding of the overall performance of the model, we will calculate some of our own diagnostics (absolute residuals and percent error).


```python
# calculate residuals: absolute value of actual - predicted
cv_results['residuals'] = abs(cv_results['y'] - cv_results['yhat'])

# calculate percent error for each observation (residual / actual)
cv_results['perc_error'] = (cv_results['residuals']/cv_results['y'])*100
cv_results.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yhat</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>y</th>
      <th>residuals</th>
      <th>perc_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>5803.000000</td>
      <td>5803.000000</td>
      <td>5803.000000</td>
      <td>5803.000000</td>
      <td>5803.000000</td>
      <td>5803.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>6.005236</td>
      <td>5.942715</td>
      <td>6.068314</td>
      <td>6.006750</td>
      <td>0.029046</td>
      <td>0.483605</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.056559</td>
      <td>0.066679</td>
      <td>0.056807</td>
      <td>0.047003</td>
      <td>0.027067</td>
      <td>0.451834</td>
    </tr>
    <tr>
      <td>min</td>
      <td>5.855141</td>
      <td>5.692719</td>
      <td>5.957656</td>
      <td>5.744207</td>
      <td>0.000001</td>
      <td>0.000025</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>5.964080</td>
      <td>5.898637</td>
      <td>6.022427</td>
      <td>5.973745</td>
      <td>0.010185</td>
      <td>0.169779</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>5.989129</td>
      <td>5.935677</td>
      <td>6.054272</td>
      <td>5.999882</td>
      <td>0.021944</td>
      <td>0.365401</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>6.052221</td>
      <td>5.994370</td>
      <td>6.108527</td>
      <td>6.039474</td>
      <td>0.040044</td>
      <td>0.665972</td>
    </tr>
    <tr>
      <td>max</td>
      <td>6.391012</td>
      <td>6.329964</td>
      <td>6.455271</td>
      <td>6.320145</td>
      <td>0.278496</td>
      <td>4.780750</td>
    </tr>
  </tbody>
</table>
</div>



From the summary table above, we can derive the following:
- The forecasts differed from the actual observations by an average of ~0.03 (residual), or an average percent error of 0.48% 
- The best prediction only missed the actual observation by 0.000025%, whereas the worst prediction missed by 4.78% (mentioned in the MAPE chart above)

### Conclusions

The performance of our model is pretty strong considering our out of box implementation of Prophet. While the performance is indicative of Prophet's power and simplicity, it should also be attributed to the dataset's low variance and volatility. The largest percentage change between two consecutive scores is less than 4%, and the standard deviation of the time series is 0.05 points from the mean.   

The objective of this analysis was to explore the Hedonometer Index for any trends and seasonalities, including the aforementioned holiday effect. Through our analysis, we have confirmed that the holiday effect has had a positive impact on the Hedonometer Index. Additionally, we discovered the "weekend-bump", a weekly seasonality within the time series. Utilizing decomposition, we identified the various trends within the time series, including the current decline. Our model forecasts that Twitter sentiment will continue falling to approximately 5.9 with a MAPE of less than 1%.  

Prophet enabled us to confirm the trends and seasonalities we discovered throughout the EDA process, in addition to forecasting, plotting, and evaluating its performance in a concise and interpretable package. Although Prophet provides significant value through its ease of use, it has a robust set of parameters that can be adjusted to further improve a model's performance. Additional improvements and next steps to this analysis could include exploring and adjusting the more advanced parameters of Prophet, as well as examining the relationship between happiness (Hedonometer Index) and the stock market (DJIA or S&P 500). 

I hope this analysis served you well as an introduction into exploratory data analysis, time series forecasting, and Facebook's Prophet library. If this time series forecast was of interest, I encourage you to perform a forecast of your own. You can explore Prophets' documentation here https://facebook.github.io/prophet/. Any comments or questions are welcomed and appreciated!

__Thank you for reading!__

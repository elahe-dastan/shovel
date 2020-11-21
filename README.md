# shovel
This repository contains my data mining course homeworks.
The homeworks of this course seem useful and interesting so I decided to tell you what is going on so you can practice too

# HW1
There is a file named covid.csv that contains information about people suffering from COVID-19 in south korea.<br/>
1. I read this file using **pandas** library
2.  This dataset is small and contains only 176 records, these are the columns

|id     |  sex   |birth_year |country  | region  |infection_reason |infected_by |confirmed_date | state  |
|-------|:------:|:---------:|:-------:|:-------:|:---------------:|:----------:|:-------------:|-------:|
|nominal| nominal| interval  | nominal | nominal |    nominal      |    ratio   |    interval   | nominal|
|-------|:------:|:---------:|:-------:|:-------:|:---------------:|:----------:|:-------------:|-------:|
| 0 nan | 0 nan  |  10 nan   | 0 nan   | 10 nan  |     81 nan      |  134 nan   |      0 nan    |  0 nan |

3. It is asked to find max, mean and std of the column birth_year<br/>
the max is 2009<br/>
Let me talk about finding mean, this column has null values and I can have different strategies facing them I calculated
the mean using two ways:<br/>
first: We can think that the null values don't exist and calculate the mean, mean() function of pandas dataframe does 
that, the mean is 1973.3855<br/>
second: I can change the pandas dataframe to a numpy array and then calculate the mean but the mean() function of numpy
array ignore the null values so I have to substitute them with a number for example zero, the mean is 1861.2613.<br/>
everything I said about mean remains the same for std. I calculated std just using pandas dataframe std function<br/>
std: 17.0328

4.yes, null values exist in the dataset.<br/>
The question is to remove the null values by a proper method but what the proper method?? if the dataset was huge and the
null values were few I would remove the records which have null values but our case is completely the opposite so I should
substitute the null values with a value. **pay attention**: sometimes a column contains null values that I don't want to 
select in feature selection or I sometimes even use a method that can handle null values but in this question we assume
that we don't want to put aside any column and our method cannot handle null values. In my opinion the best way to 
substitute null values of **numerical** columns is to put the median value instead of them **Note**: I prefer to use median
over mean cause it's more resistant to outliers. For **nominal** columns I substitute the null values with the most frequent
value.<br/>
**trouble alert**: There is column in our dataset that has date time values, it's logical to use median strategy for this
column but the SimpleImputer class that I use considers this column as string and cannot find the median I thought of two
solutions, I can find the median `df['confirmed_date'].astype('datetime64[ns]').quantile(.5)` and then use SimpleImputer
with constant strategy or I can convert the column to timestamp before passing it to SimpleImputer, the second approach 
is easier.

5. visualize data<br/>
First, I want to plot the histogram of some columns, I think the birth year and infected by columns are the most appropriate
and plotting the histogram of the other columns don't give us any information (for example id column :joy:).<br/>
Second, I'd like to plot a scatter plot so I need to choose two columns, let's find the correlation between birth_year and
confirmed_date

6. Here we like to detect and remove outliers, even if you have not studied a single book you may think of sorting or
visualizing the data and easily seen and remove outliers for datasets like the one we have in here this approach really 
works but most of the time the dataset is huge and you may prefer more automatic ways like:<br/>
1. Inter Quartile Range (IQR): Look at the code below 
```sh
import numpy as np
Q1 = np.quantile(data,0.25)
Q3 = np.quantile(data,0.75)
IQR = Q3 - Q1
```
2. Z-Score<br/>
In any distribution, about 95% of values will be within 2 standard deviations of the mean and 99.7% of the data within 3.
Based on this, any absolute value of z-score above 3 is considered as an outlier.<br/>
Z-score is calculated by substracting the mean and dividing by std<br/>
I treat with outliers like null values and substitute them with median

### linear regression
While reading the dataset using pandas I found out there are ';' instead of ',' so I wrote a bash script to solve this 
problem<br/>
1. I extract G3 as Y
2. I split the dataset to train and test
3 fit a linear regression model (as easy as a piece of cake) **Note**: we have nominal columns in our dataset which 
obviously linear regression cannot handle so we should transform these nominal values to numerical value before fitting 
the model
##### Encoding
Let me explain how I think completely, our nominal columns are school, sex, address, femsize, Pstatus, Mjob, Fjob, reason,
guardian, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic.<br/>
the categorical attributes generally fall into three groups (it's my grouping):<br/>
1. binary: they can have only two possible values we can consider one them as zero and the other as one, in our dataset 
these are binary: school, sex, address, femsize, Pstatus, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic
2. ordinal: they can have multiple categorical values but you can see an order among them for example
Like,Like Somewhat,Neutral,Dislike Somewhat,Dislike, it's obviously seen that 'Like' is much closer to 'Like Somewhat' than 
'Dislike' so the differentiate of the numbers I assign to 'Like' and 'Like Somewhat' should be less than the differentiate 
of 'Like' and 'Dislike' (you get the point!) none of our attributes is ordinal
3. nothing : they can have multiple categorical values but there is no order, in this case it's not reasonable to assign 
numerical values to the values cause the values which get closer numbers will have lower distance from each other though 
it's not correct.Mjob, Fjob, reason and guardian of our dataset are form this group **Solution**: we can use OneHotEncoding,
this strategy converts each category value into a new column and assigns a 1 or 0 (True/False) value to the column


4. Predict test data
5. Find accuracy<br/>
I use mean squared error for this purpose, the mse was 5.7495
 
 
# HW2
##### 1. Read the dataset using pandas library

![](titanic.png)

##### 2. Do something about the null values
I may change my opinion in future but with the knowledge I have right now I guess the best way is to replace the null 
values of the embarked column with the mode of the column (it has only 2 null values and replacing mode can be a good guess),
It's hard to say which column is more important in our prediction at the moment but I guess age can affect our prediction
a lot so I try not to remove the column and I use median values for the null ones but the null values of the "cabin" column
is so many that I think the column doesn't worth keeping. 

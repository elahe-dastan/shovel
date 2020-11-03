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
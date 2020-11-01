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
the mean using three ways:<br/>
first: We can think that the null values don't exist and calculate the mean, mean() function of pandas dataframe does 
that, the mean is 1973.3855<br/>
second:

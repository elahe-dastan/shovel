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
#### 1. Read the dataset using pandas library

![](titanic.png)

#### 2. Do something about the null values
I may change my opinion in future but with the knowledge I have right now I guess the best way is to replace the null 
values of the embarked column with the mode of the column (it has only 2 null values and replacing mode can be a good guess),
It's hard to say which column is more important in our prediction at the moment but I guess age can affect our prediction
a lot so I try not to remove the column and I use median values for the null ones but the null values of the "cabin" column
is so many that I think the column doesn't worth keeping. 

#### 3. Get deeper to the dataset

#### non numerical columns and decision tree
We'd like to use a decision tree to classify passengers and the decision tree classifier from sklearn library cannot work
with categorical values so we have to transform the categorical columns to numerical. I divide categorical columns into 
two groups, the ones which we can arrange in an order and the ones which we cannot. I use ordinal encoder for the first
group and one hot encoder for the second one. Now let's take a look at the categorical columns:<br/>
##### Name : 
At first I thought of dropping this column, how can names affect our prediction??!! no body stays alive because of
his/her name but it's a little trickier I found two points hidden in names first: we can find families using names and 
cause families travel together it's probable to say they all survive together or they all not, second: some words like
Miss. and Mrs. are specified in this column which gives us a sense to estimate his/her age so we can fill the null values
in the age column in a more proper way. We Asians may not be familiar with western names I've searched for it and write 
some points for you.<br/>
Look at this example:<br/>
**Baclini,Mrs.Solomon (Latifa Qurban)**<br/>
**Mrs.** indicates that she is married<br/>
**Solomon** is the name of her husband. This is an old-ish custom where wives can be referred to by therir husband names. 
For instance, if Jane Smith was married to John Smith, she could be referred to as Mrs.John Smith.<br/>
**Latifa** is her first name.<br/>
**Qurban** is her "maiden" name. This is the last name that she had before getting married.<br/>
**Baclini** is her married last name(the last name of her husband)<br/>
I take another example:<br/>
**Baclini,Miss.Marie Catherine**<br/>
Miss indicates that she is unmarried.<br/>
In this **Marie** is her first name, **Catherine** is her middle name and **Baclini** is the last name.<br/>
**Mr.** (for men of all the ages)<br/>
**Master.** (for male children)<br/>
We have other words like Dr., Sir., Col. and ... it's hard to separate each of them apart so I call all of them 
professional and I guess their age should be relatively high.<br/>
Let's sum it up I think first name and middle name have nothing to do with the passenger's survival but the last name can
help us find out families and the Mr. etc. words help us fill null ages.<br/>
**Note**: Finding last name for married women is tricky actually they have two last names and those are both important 
cause they may have a trip with their husband or parents. I may change my code in future but to make it simple I just 
consider their married last name. 
<!-- I don't separate last names for now this will make so many columns -->

Sex : There is no sex order but this column takes only two values so I prefer to use ordinal encoder over one hot encoder
which increases the number of my columns.<br/>    
Ticket : Tickets have 1. an optional string prefix and 2. a number except for the special cases Ticket='LINE. Ticket prefix 
tells you who the issuing ticket office and/or embarkation point was. Ticket number can be compared for equality that
tells you who were sharing a cabin or travelling together, or compared for closeness. The ticket = LINE have been assigned 
to a group of American Line employees for free
# Pressure time
I' m now under a lot of pressure so can't write a good readme I hope I can come back 
Embarked: I think there is no order --> one hot encoder
Embarked has 3 values 'C', 'Q', 'S'

## How to handle missing values in the test dataset???
I fill them the way I did in train dataset

## What if I see something completely new in the test dataset
In this dataset we had so many null values in the age column, to fill them I extracted the honorifics and said ...
"the average age of Mr is this and the average age of Miss is that" now what shall I do if I see the honorific Dr with
null age in the test dataset

## sklearn accuracy score
I don't know how sklearn.metrics.accuracy_score computes accuracy and have no time to check it but it's a question
92 percent accuracy :)))
# Draw the decision tree model using graphviz

# Different criterion
The default criterion of decision tree is gini I gave a try to entropy

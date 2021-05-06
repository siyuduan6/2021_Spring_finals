# 2021_Spring_finals

# The Relationships of Gun Violence, Educational Attainment, Poverty Status And Unemployment Rate in US
## Introduction
Gun violence represents a major threat to the health and safety of all Americans. According to the simple analysis from the gun violence data, the number of gun violence cases has steadily increased by more than 4 years. The argument about banning guns was always a controversial topic in the United States. Some people might say the gun should not be banned because every gun shooting should blame the person who uses the gun, not the gun itself. So we were thinking if there any correlation between the gun violence rate and the education level, poverty status, and unemployment.
## Team Members
Siyu Duan - siyud6@illinois.edu</br>
Baisheng Qiu - bqiu42@illinois.edu

## Datasets used for Analysis
Gun violence Data - https://github.com/jamesqo/gun-violence-data
Education Attainment - https://data.census.gov/cedsci/table?t=Educational%20Attainment&g=0100000US.860000&y=2017&tid=ACSST5Y2017.S1501
Unemployment - https://data.census.gov/cedsci/table?t=Employment&g=0100000US.860000&y=2017&tid=ACSST5Y2017.S0801
Poverty - https://data.census.gov/cedsci/table?t=Income%20and%20Poverty&g=0100000US.860000&y=2017&tid=ACSST5Y2017.S1701

## Hypothesis
Hypothesis 1: There is a positive correlation between the number of gun cases and unemployment rate. 

Hypothesis 2: The is a
positive correlation between the number of gun violence cases and poverty status.

Hypothesis 3: There is a negative correlation between the number of gun cases and the educational level.

Due to the imbalance population distribution between different cities, we decided to do our analysis on the zip-code level.

## Zip Code Crawling

For getting the relative zip code to specific address, we requested geolocation data from google API. And we got 61400 zip codes for clustering in analysis.

## Variable Definitions

We chose GV rate for correlation analysis and GV level for predictive analysis. GV level is based on the GV rate, and the splitting method will be mentioned later. And for eductional level, we examined two indexes with different definitions and tried to figure out which one can help improve the performance of our models. 
<p align="center">
  <img src="https://github.com/siyuduan6/2021_Spring_finals/blob/271d939ff19ac538da76bbb589fedef57fd818b7/Graphs/Variable%20definations.png">
</p> 

## Data Cleaning and Descriptions
### Missing Value 
For zip codes:

Cleaned the null values and None values on the zipcode column and the null values and empty field on the address column.
<p align="center">
  <img src="https://github.com/siyuduan6/2021_Spring_finals/blob/58f75c5ddccd6e862b22f3f08f18b5a2afcb054d/Graphs/Zip%20Code%20Clean%20(1).png">
</p> 

Cleaned irregular zip code like 27410(Too many cases, that zip code represent empty address).
<p align="center">
  <img src="https://github.com/siyuduan6/2021_Spring_finals/blob/58f75c5ddccd6e862b22f3f08f18b5a2afcb054d/Graphs/Zip%20Code%20Clean%20(2).png">
</p> 

For variables:

According to definations, we merged population, unemployment rate, poverty rate, and percentage of people with high school degree or higher and percentage of people with Bachelor's degree or higher with Gun Violence dataset together. Then we checked missing values of these variables and there are two types: NaN and "-". 

Before cleaning, there are 9847 rows.

<p align="center">
  <img src="https://github.com/siyuduan6/2021_Spring_finals/blob/main/Graphs/Before%20cleaning.jpg">
</p> 

To view the missing value, we counted missing value with two types as we mentioned. 

<p align="center">
  <img src="https://github.com/siyuduan6/2021_Spring_finals/blob/main/Graphs/Missing%20value.jpg">
</p> 

And then we viewed the skewness of all variables:

<p align="center">
  <img src="https://github.com/siyuduan6/2021_Spring_finals/blob/main/Graphs/Poverty%20dis.png">
</p> 

The distribution of poverty rate shows the right skewness.

<p align="center">
  <img src="https://github.com/siyuduan6/2021_Spring_finals/blob/main/Graphs/Unemployment%20dis.png">
</p> 

The distribution of unemployment rate shows the right skewness.

<p align="center">
  <img src="https://github.com/siyuduan6/2021_Spring_finals/blob/main/Graphs/High%20school%20dis.png">
</p> 
<p align="center">
  <img src="https://github.com/siyuduan6/2021_Spring_finals/blob/main/Graphs/Bachelor%20dis.png">
</p> 

The distribution of education level (High School or Higher) shows the left skewness and so as education level (Bachelor's or Higher).

<p align="center">
  <img src="https://github.com/siyuduan6/2021_Spring_finals/blob/main/Graphs/rate%20dis.png">
</p> 
<p align="center">
  <img src="https://github.com/siyuduan6/2021_Spring_finals/blob/main/Graphs/Rate%20log%20dis.png">
</p> 

We found the all the variables are skewed, especially GV rate, so we used Log transformation to handle the skewness of GV rate.

Since they are all numerical variables and highly skewed with only a very small number of missing values, we decided to fill the missing values by their median.

After data cleaning, we did descriptive statistics:

<p align="center">
  <img src="https://github.com/siyuduan6/2021_Spring_finals/blob/main/Graphs/Descriptive%20Statistics.png">
</p> 


## Simple Data Analyst
<p align="center">
  <img src="https://github.com/siyuduan6/2021_Spring_finals/blob/eee4e920340d9ec67daeb23b6cb4a04484858017/Graphs/%E4%B8%8B%E8%BD%BD.png">
</p> 

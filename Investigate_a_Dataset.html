# Project: Investigate a Dataset - Medical Appointment No Shows

## Table of Contents
<ul>
<li><a href="#intro">Introduction</a></li>
<li><a href="#wrangling">Data Wrangling</a></li>
<li><a href="#eda">Exploratory Data Analysis</a></li>
<li><a href="#conclusions">Conclusions</a></li>
</ul>

<a id='intro'></a>
## Introduction

### Dataset Description 

In this project we will analyze the data associated with patients who do not show for their appointments in Brazil. I will be using the "No-show appointments" dataset. This dataset includes patient id's, appointment id's, gender, scheduled day, appointment day, age, neighborhood, scholarhip, hypertension, diabetes, alcoholims, handicap, sms received, and no-show columns. 

The scheduled day column is the date the appointment was scheduled, while the appointment day column is the day of the actual appointment.

Neighborhood is the location of where the appointment will take place.

The scholarship column has a yes or no if the patient was enrolled in the "Bolsa Familia" social welfare program.

the sms received column is if the patient was sent 1 or more text messages.

In the no-show column there is a "Yes" if the patient did not show, and a "No" if the patient did show up.
I will be changing these labels.

All other columns are self-explanatory.

All cells labled with a 0 or 1; 0 is for "No" and 1 is for "Yes".




### Question(s) for Analysis

The question for analysis is, "What is the most likely reason patients miss their appointments?".

And, we will also find out which factors play an important role in this prediction.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl

%matplotlib inline
```

<a id='wrangling'></a>
## Data Wrangling


### General Properties


First, I will load in the dataset I chose using Pandas.


```python
## Here I load the dataset using the read_csv function

df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')
```

Then I would like to view the what some of the data looks like; how it is formatted, column names,etc. I will do this using the head function.


```python
## df.head() function prints out the first 5 rown in a dataset

df.head()
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
      <th>PatientId</th>
      <th>AppointmentID</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handcap</th>
      <th>SMS_received</th>
      <th>No-show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29872499824296</td>
      <td>5642903</td>
      <td>F</td>
      <td>2016-04-29T18:38:08Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>558997776694438</td>
      <td>5642503</td>
      <td>M</td>
      <td>2016-04-29T16:08:27Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4262962299951</td>
      <td>5642549</td>
      <td>F</td>
      <td>2016-04-29T16:19:04Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>867951213174</td>
      <td>5642828</td>
      <td>F</td>
      <td>2016-04-29T17:29:31Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8841186448183</td>
      <td>5642494</td>
      <td>F</td>
      <td>2016-04-29T16:07:23Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



Then I would like to know how many rows, and columns there are in total in this dataset. I wil ldo this by using the shape function.


```python
## the shape function prints out the amoutn of rows, and columns in ths format: (rows, columns).

df.shape
```




    (110527, 14)



Next, I will look aat some statistics about the datset. Such as, the mean age, and unique items ine each column. I will do this by using the describe function.


```python
## Here I use the describe function with the include parameter to show all columns.

df.describe(include = 'all')
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
      <th>PatientId</th>
      <th>AppointmentID</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handcap</th>
      <th>SMS_received</th>
      <th>No-show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>110527</td>
      <td>110527</td>
      <td>110527</td>
      <td>110527</td>
      <td>110527</td>
      <td>110527</td>
      <td>110527</td>
      <td>110527</td>
      <td>110527</td>
      <td>110527</td>
      <td>110527</td>
      <td>110527</td>
      <td>110527</td>
      <td>110527</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>103549</td>
      <td>27</td>
      <td>NaN</td>
      <td>81</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>2016-05-06T07:09:54Z</td>
      <td>2016-06-06T00:00:00Z</td>
      <td>NaN</td>
      <td>JARDIM CAMBURI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>71840</td>
      <td>24</td>
      <td>4692</td>
      <td>NaN</td>
      <td>7717</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>88208</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>147496265710392</td>
      <td>5675305</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>256094920291739</td>
      <td>71296</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>39218</td>
      <td>5030230</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4172614444192</td>
      <td>5640286</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>31731838713978</td>
      <td>5680573</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>94391720898175</td>
      <td>5725524</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>55</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>999981631772427</td>
      <td>5790484</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>115</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



I want to see if there are any columns with missing or duplicated data, and decide on which columns I will be using for my analysis. I will acheive this by using the info function. 


```python
""" Here the info function is uded to print out the columns, and specific information such as; 
column name, the data type for each column, and whether or not there are any null values"""

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 110527 entries, 0 to 110526
    Data columns (total 14 columns):
     #   Column          Non-Null Count   Dtype  
    ---  ------          --------------   -----  
     0   PatientId       110527 non-null  float64
     1   AppointmentID   110527 non-null  int64  
     2   Gender          110527 non-null  object 
     3   ScheduledDay    110527 non-null  object 
     4   AppointmentDay  110527 non-null  object 
     5   Age             110527 non-null  int64  
     6   Neighbourhood   110527 non-null  object 
     7   Scholarship     110527 non-null  int64  
     8   Hipertension    110527 non-null  int64  
     9   Diabetes        110527 non-null  int64  
     10  Alcoholism      110527 non-null  int64  
     11  Handcap         110527 non-null  int64  
     12  SMS_received    110527 non-null  int64  
     13  No-show         110527 non-null  object 
    dtypes: float64(1), int64(8), object(5)
    memory usage: 11.8+ MB
    


```python
## Here I check for duplicates

sum(df.duplicated())
```




    0



Looks like all the values in the dataset are non-null values. 

I serached for duplicates and I found 27,964 dupliucates, which I will be rmoving.

I have also decided on which columns I will be removing to make my analysis easier. 

I will be removing the appointment ID, schedule day, and appointment day columns; as I do not expect these to play important factors in my analysis. I will also be removing one from the analysis as it has a negative age, as well as all tghe duplicates in the data.

I also noticed this data shows the patient id in scientific notation. I will be changing this to an integer to make it easier to read.

I wil also be making a few changes in spelling, for a couple of the labels.


### Data Cleaning


First thing I want do is remove the columns I do not plan to use in my analysis. In this case I will be removing the appointment ID, schedule day, and appointment day columns. I will do this by using the drop function.


```python
## Here I will use the drop function to remove the columns I will not be using.

df.drop(['AppointmentID', 'ScheduledDay', 'AppointmentDay'], axis=1, inplace=True)
```


```python
## Now I will print out the first 5 rows after removing the uneccessary columns.

df.head()
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
      <th>PatientId</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handcap</th>
      <th>SMS_received</th>
      <th>No-show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29872499824296</td>
      <td>F</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>558997776694438</td>
      <td>M</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4262962299951</td>
      <td>F</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>867951213174</td>
      <td>F</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8841186448183</td>
      <td>F</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



On to find the row with the negative age.


```python
## Here I sorted the value in descending order to find the negative age using the sort values function.

df.sort_values('Age', ascending=False)
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
      <th>PatientId</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handcap</th>
      <th>SMS_received</th>
      <th>No-show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>68127</th>
      <td>31963211613981</td>
      <td>F</td>
      <td>115</td>
      <td>ANDORINHAS</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>63915</th>
      <td>31963211613981</td>
      <td>F</td>
      <td>115</td>
      <td>ANDORINHAS</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>76284</th>
      <td>31963211613981</td>
      <td>F</td>
      <td>115</td>
      <td>ANDORINHAS</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>97666</th>
      <td>748234579244724</td>
      <td>F</td>
      <td>115</td>
      <td>SÃO JOSÉ</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>No</td>
    </tr>
    <tr>
      <th>63912</th>
      <td>31963211613981</td>
      <td>F</td>
      <td>115</td>
      <td>ANDORINHAS</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2969</th>
      <td>452418952997</td>
      <td>F</td>
      <td>0</td>
      <td>RESISTÊNCIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2970</th>
      <td>4442457866355</td>
      <td>M</td>
      <td>0</td>
      <td>RESISTÊNCIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>56706</th>
      <td>51822675643792</td>
      <td>M</td>
      <td>0</td>
      <td>ILHA DO PRÍNCIPE</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>101311</th>
      <td>4419954231623</td>
      <td>M</td>
      <td>0</td>
      <td>DO MOSCOSO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>No</td>
    </tr>
    <tr>
      <th>99832</th>
      <td>465943158731293</td>
      <td>F</td>
      <td>-1</td>
      <td>ROMÃO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>110527 rows × 11 columns</p>
</div>



Now to remove the row with negative value.


```python
## Here I drop the row with the negative value for age.

df.drop(99832, axis=0, inplace=True)
```

We now see the row with the negative value is gone.


```python
## and here I sort it one more time to see if the negative value was dropped, and it was.

df.sort_values('Age', ascending=False)
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
      <th>PatientId</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handcap</th>
      <th>SMS_received</th>
      <th>No-show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>68127</th>
      <td>31963211613981</td>
      <td>F</td>
      <td>115</td>
      <td>ANDORINHAS</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>63915</th>
      <td>31963211613981</td>
      <td>F</td>
      <td>115</td>
      <td>ANDORINHAS</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>76284</th>
      <td>31963211613981</td>
      <td>F</td>
      <td>115</td>
      <td>ANDORINHAS</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>97666</th>
      <td>748234579244724</td>
      <td>F</td>
      <td>115</td>
      <td>SÃO JOSÉ</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>No</td>
    </tr>
    <tr>
      <th>63912</th>
      <td>31963211613981</td>
      <td>F</td>
      <td>115</td>
      <td>ANDORINHAS</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>27550</th>
      <td>3791163514832</td>
      <td>M</td>
      <td>0</td>
      <td>MARUÍPE</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>27557</th>
      <td>982799219678115</td>
      <td>F</td>
      <td>0</td>
      <td>SANTA CECÍLIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>27558</th>
      <td>56727644655921</td>
      <td>F</td>
      <td>0</td>
      <td>MARUÍPE</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>98247</th>
      <td>364724578963742</td>
      <td>F</td>
      <td>0</td>
      <td>JABOUR</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>92096</th>
      <td>21494487427167</td>
      <td>F</td>
      <td>0</td>
      <td>ENSEADA DO SUÁ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>110526 rows × 11 columns</p>
</div>



Next I will be removing all the duplicates.


```python
## Here U use the drop duplicates function to remove all the duplicates

df.drop_duplicates(inplace=True)
```

We now have 82,562 entries, as all the duplicates have been removed


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 82562 entries, 0 to 110524
    Data columns (total 11 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   PatientId      82562 non-null  float64
     1   Gender         82562 non-null  object 
     2   Age            82562 non-null  int64  
     3   Neighbourhood  82562 non-null  object 
     4   Scholarship    82562 non-null  int64  
     5   Hipertension   82562 non-null  int64  
     6   Diabetes       82562 non-null  int64  
     7   Alcoholism     82562 non-null  int64  
     8   Handcap        82562 non-null  int64  
     9   SMS_received   82562 non-null  int64  
     10  No-show        82562 non-null  object 
    dtypes: float64(1), int64(7), object(3)
    memory usage: 7.6+ MB
    

Now I will convert the patient ID, from scientific notation to an integer.


```python
## Here is how I changed the scientific notation to an integer using the options function.

pd.options.display.float_format = '{:.0f}'.format

```


```python
## Here is the first 5 rows with the patient ID as an integer.

df.head()
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
      <th>PatientId</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handcap</th>
      <th>SMS_received</th>
      <th>No-show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29872499824296</td>
      <td>F</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>558997776694438</td>
      <td>M</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4262962299951</td>
      <td>F</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>867951213174</td>
      <td>F</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8841186448183</td>
      <td>F</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



Now I will some changes in spelling, for the labels. Specfically; neighbourhood, patient id, hyperstension, and handicap. I will do this using the reanme function.


```python
## Here I will be changing the column names using the rename function.

df.rename(columns={'PatientId':'PatientID', 'Neighbourhood':'Neighborhood', 'Hipertension':'Hypertension', 'Handcap':'Handicap'}, inplace=True)
```


```python
## Now I will be changing the "Yes" and "No" in the "No-show" column to 'Showed' and "Did not show".

df['No-show'].replace({'No': 'Showed', 'Yes': 'Missed'}, inplace=True)
```


```python
## Here are the column name changes. 

df.head()
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
      <th>PatientID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Neighborhood</th>
      <th>Scholarship</th>
      <th>Hypertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SMS_received</th>
      <th>No-show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29872499824296</td>
      <td>F</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Showed</td>
    </tr>
    <tr>
      <th>1</th>
      <td>558997776694438</td>
      <td>M</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Showed</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4262962299951</td>
      <td>F</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Showed</td>
    </tr>
    <tr>
      <th>3</th>
      <td>867951213174</td>
      <td>F</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Showed</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8841186448183</td>
      <td>F</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Showed</td>
    </tr>
  </tbody>
</table>
</div>



<a id='eda'></a>
## Exploratory Data Analysis


In this section I will explore the data set and observe any trends in the data. We will also try and predict if a patient will miss an appointment, or not. We will do this by analyzing different variables in the dataset.

At first look I see, in the fgure below, may not play a role in patients missing their appointments.


```python
## Here I used a basic histogram function to plot all the variables in the dataset.

df.hist(alpha=.5, rwidth=1, ec='black', figsize=(10,8));
plt.suptitle('Medical Appointment No-Shows', x=0.5, y=1, fontsize='xx-large');
```


    
![png](output_38_0.png)
    


### Is Age A Factor For Patients Missing Appointments?

Here I will be doing an analysis on age, and if it has any effect on whether pateints miss their appointments.

In this fist figure I see that more users attend their appointments.

Although I do see that the younger crowd, mostly patients under the age 40 are most likely to miss their appointment.


```python
## Here I find the mean age for the dataframe

print('The average age is:', int(np.mean(df['Age'])), 'years old.')
```

    The average age is: 36 years old.
    


```python
"""Here I used another histogram to make two plots, both grouped by age, one for patients who attended their appointments 
and onefor patients who missed their appointment"""

df.hist(column='Age', by='No-show', bins=35, alpha=.5, rwidth=1, ec='black', figsize=(15,5));
plt.suptitle('Patients Who Attend And Do Not Attend Appointments By Age', x=0.5, y=1.05, fontsize='xx-large');
plt.xlabel('Age');
plt.ylabel('Appointments');
```


    
![png](output_41_0.png)
    


### Is Gender A Factor For Patients Missing Appointments?

In this figure I analyze if gender is a factor for patients missing their appointments.

Although, females make up for most of the data I see that more females miss their appointment than their male counterparts, as pictured in the "Yes" histogram; it is also clear that more females attend their appointments than males.

Gender may be a factor.


```python
""""Her I made two histograms. One for both females, and males that attended their appointments; 
and one for males and females that missed their appointments"""

df.hist(column='Gender', by='No-show', bins=4, alpha=.5, rwidth=1, ec='black', figsize=(15,5));
plt.suptitle('Appointments By Gender', x=0.5, y=1.05, fontsize='xx-large');
plt.xlabel('Gender');
plt.ylabel('Appointments');
```


    
![png](output_43_0.png)
    


### Creating A Function For A Pie Chart


```python
''' I know I will be using a lot of pie charts in this analysis, so I will be creating a function that creates
a pie chart using a variable as the parameter.'''

def medpie(col):
    explode = (0.25, 0.0);
    df.groupby(['No-show']).sum().plot(kind='pie', y=col, autopct='%1.0f%%', 
                                  colors = ['green', 'red'], figsize=(10,8), shadow=True, explode=explode);
    plt.title('Appointments For Patients With '+col, fontsize='xx-large',x=0.5, y=1.05);
```

### Is The Welfare Program A Factor For Patients Missing Appointments?


```python
''' Here I use a pie chart to see the percentage of patients in the Welfare Program 'Bolsa Familia',
that miss or attend their appointments'''

medpie('Scholarship')
```


    
![png](output_47_0.png)
    


As we can see here 73% of patients in the program do attend their appointments.

### Is Hypertension A Factor Why Patients Miss Their Appointments?


```python
''' Here I use the function I created to see the percentage of patients with hypertension,
that miss or attend their appointments'''

medpie('Hypertension')
```


    
![png](output_50_0.png)
    


We can see here that 80% of the patients with hypertension attend their meetings. 

### Is Diabetes A Factor Why Patients Miss Their Appointments?


```python
''' Here I use a pie chart to see the percentage of patients with diabetes,
that miss or attend their appointments'''


medpie('Diabetes')
```


    
![png](output_53_0.png)
    


And once again we see that diabetes is not a factor, as 79% of the patients with diabetes attend their appointments.

### Is Alcoholism A Factor Why Patients Miss Their Appointments?


```python
''' Here I use a pie chart to see the percentage of patients with diabetes,
that miss or attend their appointments'''


medpie('Alcoholism')
```


    
![png](output_56_0.png)
    


So, far the lowest attendance rate is patients with alcoholism at 72%. Here we can see that most patients with alcoholism still attend their meetings.

### Is Having A Handicap  A Factor Why Patients Miss Their Appointments?


```python
''' Here I use a pie chart to see the percentage of patients with diabetes,
that miss or attend their appointments'''


medpie('Handicap')
```


    
![png](output_59_0.png)
    


As we can see in this pie chart, 79% percent of patients with a handicap attend their meetings. 

<a id='conclusions'></a>
## Conclusions

So, the most likely reason patients miss their appointments is if they suffer from alcoholism. Only 72% of patients who received a reminder for their appointment attended their appointment. Coming in at a close second would be if the patient suffers from diabetes, and or a handicap at 79%. 

The data could be further explored to try and see if a patients will miss an appointment  can be predicted. Maybe using a combination of factors, and some machine learning.

### Limitation

The limitations for this analysis would be that we are working with sample data, as the population is Brazil is far greater than the sample in this analysis.

Also, the data may not be as accurate as proposed. Some patients may not enter correct information when seen.

## Submitting your Project 



## OPTIONAL: Question for the reviewer
 
If you have any question about the starter code or your own implementation, please add it in the cell below. 

For example, if you want to know why a piece of code is written the way it is, or its function, or alternative ways of implementing the same functionality, or if you want to get feedback on a specific part of your code or get feedback on things you tried but did not work.

Please keep your questions succinct and clear to help the reviewer answer them satisfactorily. 

> **_Your question_**


```python
from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])
```




    0



#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets 
from ipywidgets import interact, interactive,fixed,interact_manual
import plotly.express as px


# In[2]:


df=pd.read_csv('stage3.csv') ##
zip_code=pd.read_csv('zip_code_crawler.csv')
df['year']=pd.DatetimeIndex(df['date']).year
df['month']=pd.DatetimeIndex(df['date']).month
year=df.groupby('year').size()
plt.plot(year, linestyle = 'dotted')
plt.show()

df[df['date']=='2015-07-04'].count()

df=df[['incident_id','date','state','city_or_county','n_killed',
                   'n_injured','year','participant_gender','participant_status','month','incident_characteristics']]
## We only need few columns in below analyst, so extract those columns we need.


df_2015=df[df['year']==2015]
mon_2015=df_2015.groupby('month').size()
plt.plot(mon_2015, linestyle = 'dotted')
plt.title('Cases by Month in 2015')
plt.show()


df_2016=df[df['year']==2016]
mon_2016=df_2016.groupby('month').size()
plt.plot(mon_2016, linestyle = 'dotted')
plt.title('Cases by Month in 2016')
plt.show()


df_2017=df[df['year']==2017]
mon_2017=df_2017.groupby('month').size()
plt.plot(mon_2017, linestyle = 'dotted')
plt.title('Cases by Month in 2017')
plt.show()


def july4(year):
    """
    This function print out the top 10 number of incidents in eahc year
    :param year: The dataframe from the specific year
    >>> df_2014=df[df['year']==2014]
    >>> df_2013=df[df['year']==2013]
    >>> july4(df_2014)
    date
    2014-09-06    220
    2014-01-01    216
    2014-07-05    212
    2014-10-25    210
    2014-09-11    203
    2014-07-13    200
    2014-08-01    200
    2014-08-17    199
    2014-07-20    199
    2014-07-29    193
    dtype: int64
    >>> july4(df_2013)
    date
    2013-06-23    5
    2013-08-18    4
    2013-10-27    4
    2013-07-07    4
    2013-09-17    4
    2013-07-13    4
    2013-08-25    4
    2013-01-01    3
    2013-06-02    3
    2013-06-15    3
    dtype: int64


    """
    print(year.groupby('date').size().sort_values(ascending=False).head(10))

july4(df_2015)
july4(df_2016)
july4(df_2017)



year_df=df.groupby('year').size().reset_index()
year_df.drop([0],inplace=True)
year_df.drop([5],inplace=True)
year_df.columns=['Year','Count']

fig = plt.figure(figsize =(10, 7))
 
# Horizontal Bar Plot
plt.bar(year_df['Year'], year_df['Count'])
plt.locator_params(integer=True)
# Show Plot
plt.title('The number of gun violence incidents by Year',fontsize=20)
plt.show()


# In[13]:


df.isnull().sum()
## There are many missing values in the participant's info columns, but we are not going to clean those columns
## becuase each case might have more than one particpant.Using mode to replace missing values is not appropriate in this case


# In[14]:


df.groupby('state').size().sort_values(ascending=False).head(10)

#The most dangerous states between 2013 to 2018


# In[19]:


year_list=df.groupby('year').size().index

## The list that contain all the years.

@ipywidgets.interact(x=year_list)
def hist_plot(x):
    """
    This function is to plot the number of cases by state in different years
    :param x:The year list,contain the number of year.

    """
    top3=df[df['year']==x].groupby('state').size().sort_values(ascending=False).head(3).index
    fig_dims = (16, 14)
    fig, ax = plt.subplots(figsize=fig_dims)
    ax = sns.countplot(x=df[df['year']==x]['state'],data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
    plt.title( 'Three States have most gun violence incidents : '+top3[0]+','+top3[1]+','+top3[2],fontsize=25)
    plt.suptitle('The number of cases by State in ' +str(x),fontsize=25)
    plt.xlabel('States',fontsize=15,fontweight='bold')
    plt.ylabel('Number of cases',fontsize=15)
    plt.show()

## obviously we can few states such as California , Illinois and Florida have  much higher number of cases than other States. 


# In[20]:


px.scatter(df,x='n_injured',y='n_killed',size='year')
## large number of injured usually coming with the large number of killed.
## Maybe those cases was recorded from mass shooting cases.


# In[21]:


def count_gender(column):
    """
    This function count the proportion of gender in all cases between 2013-2018
    and plot it as histogram
    :param column:The columns the gender
    :return:statists of gender in dataframe format
    """
    male=0
    female=0
    for line in column:
        if 'Male' or 'Female' in str(line):
            count_male=str(line).count('Male')
            male+=count_male
            count_female=str(line).count('Female')
            female+=count_female
            data = [['Male', male], ['female', female]]
    gender_data = pd.DataFrame(data, columns = ['Gender', 'count'])
    fig, ax = plt.subplots(1,1, figsize = (15,9))

    x = gender_data['Gender']
    y = gender_data['count']

    ax.bar(x, y)
    ax.set_title('Number of count in gun violence cases - By Sex', fontsize = 18)
    plt.xlabel('Sex', fontsize = 15)
    plt.ylabel('count', fontsize = 15)
    plt.show()
    return gender_data


# In[22]:


count_gender(df['participant_gender'])


# In[24]:


def count_type(column):
    """
    This function is to count the incident_characteristics
    :param column: The column of incident_characteristics
    >>> count_type(df['incident_characteristics'])
    1637
    17988
    5678
    """
    mass=0
    officer=0
    gang=1
    for line in column:
        if 'Officer'in str(line):
            officer+=1
        if 'Mass Shooting'in str(line):
            mass+=1
        if 'Gang'in str(line):
            gang+=1
            
    print(mass)
    print(officer)
    print(gang)
    


# In[25]:





# In[27]:


def count_status(column):
    """
    This function count the status of incidents
    :param column:The column of incidents'status 
    :return: The statistics of incident status in dataframe format
    """
    arrested=0
    killed=0
    injured=0
    unharmed=0
    for line in column:
        if 'Injured' or 'Arrested'or'killed'or'Unharmed' in str(line):
            count_arrest=str(line).count('Arrested')
            arrested+=count_arrest
            
            count_killed=str(line).count('Killed')
            killed+=count_killed
            
            count_injured=str(line).count('Injured')
            injured+=count_injured
            
            count_unharmed=str(line).count('Unharmed')
            unharmed+=count_unharmed
    total=arrested+killed+injured+unharmed
    status = [['Killed', killed,killed/total*100], ['Arrested', arrested,arrested/total*100],
              ['Injured', injured,injured/total*100],['Unharmed', unharmed,unharmed/total*100]]
    status_data = pd.DataFrame(status, columns = ['Status', 'Count','percentage'])
    
    print("total cases",total)
    
    
    labels = status_data['Status']
    sizes = status_data['percentage']
    explode = (0.2,0,0,0)

    fig1, ax1 = plt.subplots(figsize = (15,9))

    wedges, texts, autotexts =  ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)

    ax1.axis('equal')

    ax1.legend(wedges, labels,
              title="Age Groups",
              loc="center left")

    plt.setp(autotexts, size = 15, weight = "bold") 
    ax1.set_title("Participant status", fontsize = 18)

    plt.show()
    
    return status_data
        


# In[32]:


count_status(df['participant_status'])


# In[33]:


state=df.groupby('state').size()
state_df=state.reset_index()
state_df.columns = ['State', 'count']


# In[43]:


us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}
state_df['abbrev'] = state_df['State'].map(us_state_abbrev)
state_df


# In[44]:


def incident_map(data):
    """
    This function plot the choropleth maps of incidents by state.
    :param data:Number of incidents counts by states in dataframe
    """
    fig = px.choropleth(data,  # Input Pandas DataFrame
                        locations="abbrev",  # DataFrame column with locations
                        color="count",
                        color_continuous_scale='spectral_r',# DataFrame column with color values
                        hover_name="abbrev", # DataFrame column hover info
                        locationmode = 'USA-states',
                        scope='usa') # Set to plot as US States
    fig.add_scattergeo(
        locations=state_df['abbrev'],
        locationmode='USA-states',
        text=state_df['abbrev'],
        mode='text')
    fig.update_layout(title='The most number of gun violence incidents by States')


    fig.show()


# In[46]:


incident_map(state_df)


# In[17]:


zip_2017=zip_code
final_2017=zip_2017[['incident_id','date','address','state','city_or_county','n_killed','n_injured','zip']]


# In[18]:


final_2017.isnull().sum()
# we found there are 1771 zip code is null values, let's check it out


# In[19]:


final_2017.groupby('zip').size().sort_values(ascending=False).head(10)
## The zip code number 27410 have too many cases, this is not normal. Let's check it out
## But first we may deal with the nan, None values first


# In[20]:


zip_clean=final_2017[(final_2017['zip']!='None')& (final_2017['zip'].notnull())]
zip_clean
# we selected the rows that didn't contain null values and "None" on zipcode


# In[21]:


zip_clean[zip_clean['address'].isnull()]

# Now we knew the columns who doesn't have addresss will output the zipcode of 27410
# we decided to remove these rows since the zipcodes are incorrect may affect the result of our analyst


# In[50]:


zip_clean=zip_clean[zip_clean['address'].notnull()]
zip_clean.groupby('zip').size().sort_values(ascending=False)
## The zip code looks more better , but some of them have decimal point 


# In[51]:


zip_clean[pd.to_numeric(zip_clean['zip'], errors='coerce').isnull()]

## remove those rows too


# In[52]:


zip_clean.drop([13315],inplace=True)
zip_clean.drop([20071],inplace=True)


# In[53]:


zip_clean['zip']=zip_clean['zip'] = zip_clean['zip'].astype(str).replace('\.0', '', regex=True)
## we corrected the rows who has decimal point in zipcode. we didn't remove the rows because we found the zip code was correct base on their address.


# In[54]:


zip_clean.isnull().sum()
## Now the data was cleaned


# In[513]:


zip_clean
# This dataframe will output as "IS597_zip_cleaned.csv"
# please continue to look up other py files


# In[ ]:





# In[ ]:





from scipy import stats
import pandas as pd
import seaborn as sns
from statsmodels.api import OLS
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


# GV_rate = the number of GV cases / the population of the region /1000)
# the number of case per 1000 people
# edu = the percentage of people over 25 years old with high school or higher degree
# edu2 = the percentage of people over 25 years old with Bachelor's or higher degree
# pov = the percentage of people who are under poverty level
# un = the percentage of people over 16 years old who are unemployed

def basic_analysis(file_name):
    """
    Do some basic analysis and plots with the dataset
    :param file_name: Dataset that contains the case data
    :return: DataFrame with columns for analysis
    >>> basic_analysis('gv_2013-2018.csv')
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
    date
    2015-01-01    214
    2015-07-04    211
    2015-07-05    209
    2015-05-17    205
    2015-05-26    201
    2015-05-25    201
    2015-08-01    197
    2015-09-27    197
    2015-09-20    196
    2015-08-02    195
    dtype: int64
    total cases 468589
            incident_id  ...                           incident_characteristics
    0            461105  ...  Shot - Wounded/Injured||Mass Shooting (4+ vict...
    1            460726  ...  Shot - Wounded/Injured||Shot - Dead (murder, a...
    2            478855  ...  Shot - Wounded/Injured||Shot - Dead (murder, a...
    3            478925  ...  Shot - Dead (murder, accidental, suicide)||Off...
    4            478959  ...  Shot - Wounded/Injured||Shot - Dead (murder, a...
    ...             ...  ...                                                ...
    239672      1083142  ...                          Shots Fired - No Injuries
    239673      1083139  ...  Shot - Dead (murder, accidental, suicide)||Ins...
    239674      1083151  ...                             Shot - Wounded/Injured
    239675      1082514  ...          Shot - Dead (murder, accidental, suicide)
    239676      1081940  ...  Shot - Dead (murder, accidental, suicide)||Sui...
    <BLANKLINE>
    [239677 rows x 11 columns]

    """
    df = pd.read_csv(file_name)  # Import all the case data
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    df = df[['incident_id', 'date', 'state', 'city_or_county', 'n_killed',
             'n_injured', 'year', 'participant_gender', 'participant_status', 'month', 'incident_characteristics']]

    incideng_year(df)  # Plot the number of gun violence incidents by Year
    for year in [2013, 2014, 2015, 2016, 2017]:
        showcase(df, year)  # Show the number of cases by month in year 2013-2017
        hist_plot(df, year)  # Show the number of cases by state in year 2013-2017

    df_2013 = df[df['year'] == 2013]
    df_2014 = df[df['year'] == 2014]
    df_2015 = df[df['year'] == 2015]
    for d in [df_2013, df_2014, df_2015]:
        july4(d)  # Print out the top 10 days with the highest numbers of incidents in 2013, 2014, 2015

    px.scatter(df, x='n_injured', y='n_killed', size='year')  # Plot the injuries vs kills by year
    count_status(df['participant_status'])  # Show the status of all the gun violence cases
    # state_data = set_state(df) # Convert the state name to abbreviation
    # incident_map(df, state_data) # Plot the most number of gun violence incidents by States
    return df


def showcase(df, year):
    """
    This showcase shows the number of cases by month in particular year
    :param df: Dataframe that contains cases data
    :param year: The year when the cases happened
    :return:
    >>> df = pd.read_csv('gv_2013-2018.csv')
    >>> df['year'] = pd.DatetimeIndex(df['date']).year
    >>> df['month'] = pd.DatetimeIndex(df['date']).month
    >>> showcase(df, 2014)  #doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]
    """
    df_year = df[df['year'] == year]
    mon_year = df_year.groupby('month').size()
    pic = plt.plot(mon_year, linestyle='dotted')
    plt.title('Cases by Month in' + str(year))
    plt.show()
    return pic


def july4(df):
    """
    This function print out the top 10 days with the highest numbers of incidents in each year
    :param df: The dataframe from the specific year
    >>> df = pd.read_csv('gv_2013-2018.csv')
    >>> df['year'] = pd.DatetimeIndex(df['date']).year
    >>> df['month'] = pd.DatetimeIndex(df['date']).month
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
    print(df.groupby('date').size().sort_values(ascending=False).head(10))


def incideng_year(data):
    """
    Plot the number of gun violence incidents by Year
    :param data: Dataframe that contains cases data
    :return:
    >>> df = pd.read_csv('gv_2013-2018.csv')
    >>> df['year'] = pd.DatetimeIndex(df['date']).year
    >>> df['month'] = pd.DatetimeIndex(df['date']).month
    >>> incideng_year(df)
    <Figure size 1000x700 with 1 Axes>

    """
    year_df = data.groupby('year').size().reset_index()
    year_df.drop([0], inplace=True)
    year_df.drop([5], inplace=True)
    year_df.columns = ['Year', 'Count']
    fig = plt.figure(figsize=(10, 7))
    # Horizontal Bar Plot
    plt.bar(year_df['Year'], year_df['Count'])
    plt.locator_params(integer=True)
    # Show Plot
    plt.title('The number of gun violence incidents by Year', fontsize=20)
    plt.show()

    return fig


def hist_plot(df, year):
    """
    This function is to plot the number of cases by state in different years
    :param x:The year list,contain the number of year.
    >>> df = pd.read_csv('gv_2013-2018.csv')
    >>> df['year'] = pd.DatetimeIndex(df['date']).year
    >>> df['month'] = pd.DatetimeIndex(df['date']).month
    >>> hist_plot(df, 2017)
    <AxesSubplot:title={'center':'Three States have most gun violence incidents : Illinois,California,Florida'}, xlabel='States', ylabel='Number of cases'>

    """
    top3 = df[df['year'] == year].groupby('state').size().sort_values(ascending=False).head(3).index
    fig_dims = (16, 14)
    fig, ax = plt.subplots(figsize=fig_dims)
    ax = sns.countplot(x=df[df['year'] == year]['state'], data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
    plt.title('Three States have most gun violence incidents : ' + top3[0] + ',' + top3[1] + ',' + top3[2], fontsize=25)
    plt.suptitle('The number of cases by State in ' + str(year
                                                          ), fontsize=25)
    plt.xlabel('States', fontsize=15, fontweight='bold')
    plt.ylabel('Number of cases', fontsize=15)
    plt.show()
    return ax


def count_status(column):
    """
    This function count the status of incidents
    :param column:The column of incidents'status
    :return: The statistics of incident status in dataframe format

    >>> df = pd.read_csv('gv_2013-2018.csv')
    >>> df['year'] = pd.DatetimeIndex(df['date']).year
    >>> df['month'] = pd.DatetimeIndex(df['date']).month
    >>> df = df[['incident_id', 'date', 'state', 'city_or_county', 'n_killed',
    ... 'n_injured', 'year', 'participant_gender', 'participant_status', 'month', 'incident_characteristics']]
    >>> count_status(df['participant_status'])
    total cases 468589
         Status   Count  percentage
    0    Killed   60478   12.906406
    1  Arrested   99333   21.198321
    2   Injured  118395   25.266278
    3  Unharmed  190383   40.628995

    """
    arrested = 0
    killed = 0
    injured = 0
    unharmed = 0
    for line in column:
        if 'Injured' or 'Arrested' or 'killed' or 'Unharmed' in str(line):
            count_arrest = str(line).count('Arrested')
            arrested += count_arrest
            count_killed = str(line).count('Killed')
            killed += count_killed
            count_injured = str(line).count('Injured')
            injured += count_injured
            count_unharmed = str(line).count('Unharmed')
            unharmed += count_unharmed
    total = arrested + killed + injured + unharmed
    status = [['Killed', killed, killed / total * 100], ['Arrested', arrested, arrested / total * 100],
              ['Injured', injured, injured / total * 100], ['Unharmed', unharmed, unharmed / total * 100]]
    status_data = pd.DataFrame(status, columns=['Status', 'Count', 'percentage'])
    print("total cases", total)
    labels = status_data['Status']
    sizes = status_data['percentage']
    explode = (0.2, 0, 0, 0)
    fig1, ax1 = plt.subplots(figsize=(15, 9))
    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                                       shadow=True, startangle=90)
    ax1.axis('equal')
    ax1.legend(wedges, labels,
               title="Age Groups",
               loc="center left")
    plt.setp(autotexts, size=15, weight="bold")
    ax1.set_title("Participant status", fontsize=18)
    plt.show()

    return status_data


def dataclean(file):
    """
    Clean the missing and unusual zipcodes
    :param data_2017: dataset with zip codes and relative addresses
    :return: cleaned DataFrame
    """
    data_2017 = pd.read_csv(file)
    print(data_2017.isnull().sum())  # check null values out
    data_2017.groupby('zip').size().sort_values(ascending=False).head(
        10)  # Check zip code area with unusual number of cases
    zip_clean = data_2017[(data_2017['zip'] != 'None') & (data_2017['zip'].notnull())]
    # we selected the rows that didn't contain null values and "None" on zipcode
    zip_clean[zip_clean['address'].isnull()]
    # There are columns without address,
    # so remove these rows since the zipcodes are incorrect may affect the result of our analyst
    zip_clean = zip_clean[zip_clean['address'].notnull()]
    zip_clean.groupby('zip').size().sort_values(
        ascending=False)
    # Drop the weird zip codes like 13315, 20071
    zip_clean[pd.to_numeric(zip_clean['zip'], errors='coerce').isnull()]
    zip_clean.drop([13315], inplace=True)
    zip_clean.drop([20071], inplace=True)
    # Correct the rows who has decimal point in zipcode
    zip_clean['zip'] = zip_clean['zip'] = zip_clean['zip'].astype(str).replace('\.0', '',
                                                                               regex=True)
    # Print cleaned data
    print(zip_clean.isnull().sum())
    zip_clean.to_csv("GV_zip_cleaned.csv")
    return zip_clean


def zip_split(file):
    """
    Split the zip code from address.
    :param file: dataframe containing variables
    :return: file with new column:zip

    >>> file1 = pd.DataFrame({"Geographic Area Name": ["ZCTA5 35004","ZCTA5 35005","ZCTA5 35006"]})
    >>> print(zip_split(file1))
      Geographic Area Name    zip
    0          ZCTA5 35004  35004
    1          ZCTA5 35005  35005
    2          ZCTA5 35006  35006

    """
    file["zip"] = file["Geographic Area Name"].str.split(" ", expand=True)[1].astype("str")
    return file


def count_inf_none(db, column_names):
    """
    Find whether there are NAN values in each column and handle them.
    :param db: dataset you what to deal with
    :param column_names: the columns you want to deal with
    :return: a dataset without NAN values

    >>> file2 = pd.DataFrame({"a": [35004,35005,35006, "-", 35006], "b":[5, 2.5,8,6, np.inf]})
    >>> print(count_inf_none(file2, ["a", "b"]))
    a 0
    a 1
    b 1
    b 0
             a    b
    0  35004.0  5.0
    1  35005.0  2.5
    2  35006.0  8.0
    """
    db_new = db
    for name in column_names:
        print(name, db[db[name] == np.inf][name].count())
        print(name, db[db[name] == "-"][name].count())  # Print the number of infinite values and "-"
        db_new = db_new.drop(db_new[db_new[name] == np.inf].index)
        db_new = db_new.drop(db_new[db_new[name] == "-"].index)  # Drop two kinds of values
        db_new = db_new.astype({name: "float64"})
        db_new.to_csv("Variables.csv")  # Store the values into csv file
    return db_new


def mad_method(df, variable_name):
    """
    Refered to https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-1-4ece5098b755
    Do MAD (Median Absolute Deviation) to handle skewness
    :param df: dataframe contains variables
    :param variable_name:a targeted variable
    :return:the indexes of outliers

    >>> file3 = pd.DataFrame({"a": [35004,35005,35006, 205,446,888, 35006], "b":[5, 2.5,8,6, 5,10,8]})
    >>> print(mad_method(file3, "a"))
    [3, 4, 5]
    """
    # Takes two parameters: dataframe & variable of interest as string
    med = np.median(df[variable_name])
    mad = np.abs(stats.median_absolute_deviation(df[variable_name]))  # Calculate median absolute deviation
    threshold = 3
    outlier = []
    for i, v in enumerate(df.loc[:, variable_name]):
        t = (v - med) / mad
        if t > threshold or t < -1 * threshold:
            outlier.append(i)  # Append outliers into a list
        else:
            continue
    return outlier

def draw_relation(db, v, i):
    """
    Sort the dataframe by a specific variable and draw the relationship between two variables as a barplot
    :param db: dataframe containing variables
    :param v: x variable
    :param i: y variable
    :return:
    """
    fig, ax = plt.subplots(figsize=(20, 5))
    plt.xticks(rotation=30)  # Set ticks
    sns.barplot(x=db[v], y=db[i], palette="Blues_d")
    plt.show()


def linear(X, y):
    """
    Multi-Linear Regression
    :param X: x variables
    :param y: y variable
    :return: linear regression model
    >>> db = pd.read_csv("Variables_for_analysis.csv")# doctest:+ELLIPSIS
    >>> pov = db["poverty"] / 100# doctest:+ELLIPSIS
    >>> un = db["unemployment"] / 100# doctest:+ELLIPSIS
    >>> edu = db["High school or higher"] / 100# doctest:+ELLIPSIS
    >>> xv1 = pd.DataFrame([edu, pov, un]).T# doctest:+ELLIPSIS
    >>> yv = pd.DataFrame(np.log10(db["rate"]))# doctest:+ELLIPSIS
    >>> linear(xv1, yv) # doctest:+ELLIPSIS
                                     OLS Regression Result...
    <statsmodels.regression.linear_model.RegressionResultsWrapper object at ...>

    """
    # Construct the multi-linear model
    ols = OLS(y, X).fit()
    print(ols.summary())
    return ols


def plot_correlation(x):
    """
    Plot correlation matrix
    :param x: dataframe containing x and y variables
    :return: correlation matrix
    >>> db = pd.read_csv("Variables_for_analysis.csv")
    >>> pov = db["poverty"] / 100
    >>> un = db["unemployment"] / 100
    >>> edu = db["High school or higher"] / 100
    >>> edu2 = db["Bachelor or higher"] / 100
    >>> xv1 = pd.DataFrame([edu, pov, un]).T
    >>> yv = pd.DataFrame(np.log10(db["rate"]))
    >>> dba = xv1.merge(yv, left_index=True, right_index=True)
    >>> plot_correlation(dba)
                           High school or higher   poverty  unemployment      rate
    High school or higher               1.000000 -0.699980     -0.557392 -0.314174
    poverty                            -0.699980  1.000000      0.625390  0.441839
    unemployment                       -0.557392  0.625390      1.000000  0.291978
    rate                               -0.314174  0.441839      0.291978  1.000000
    """
    corr = x.corr(method="spearman")
    plt.matshow(corr)
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )  # Draw a heatmap for correlations
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )  # Set tick labels
    return corr


if __name__ == '__main__':
    """Do the preliminary analysis"""
    basic_analysis('gv_2013-2018.csv')

    """Deal with variables"""
    # Import files
    # First clean the weird and none values in "zip" column
    dataclean('zip_code_crawler1.csv')
    # The cleaned file will be exported as a file named GV_zip_cleaned.csv
    gv = pd.read_csv("GV_zip_cleaned.csv", header=0)
    pop = pd.read_csv("pop-2017.csv", header=1, low_memory=False)
    unemployment = pd.read_csv("unemployment-2017.csv", header=1, low_memory=False)
    education = pd.read_csv("educational_attainment_2017.csv", header=1, low_memory=False)
    poverty = pd.read_csv("poverty-2017.csv", header=1, low_memory=False)
    gv_new = gv.groupby(["zip"])["incident_id"].count().to_frame()
    gv_new1 = gv.groupby(["zip"])[["n_killed", "n_injured"]].agg({"n_killed": sum, "n_injured": sum})
    gv_new["n_killed"] = gv_new1.iloc[:, 0]
    gv_new["n_injured"] = gv_new.iloc[:, 1]
    gv_new = gv_new.reset_index()
    # Transform the format of zip code
    gv_new["zip"] = gv_new["zip"].astype("str")
    gv_new = gv_new.rename(columns={"incident_id": "cases"})
    k = 0
    while k < gv_new.shape[0]:
        if len(gv_new.loc[k, "zip"]) == 4:
            gv_new.loc[k, "zip"] = "0" + gv_new.loc[k, "zip"]
        k += 1
    # Extract variables
    gv_new = gv_new.merge(zip_split(pop), on="zip")
    gv_new["population"] = gv_new["Estimate!!Total!!Total population"]
    gv_new["rate"] = gv_new["cases"] / (gv_new["population"] / 1000)
    gv["zip"] = gv["zip"].astype("str")
    gv_new["city"] = gv_new.merge(gv, on="zip")["city_or_county"]
    un_new = zip_split(unemployment)[["zip",
                                      "Estimate!!Total!!Population 16 years and over",
                                      "Estimate!!Unemployment rate!!Population 16 years and over"]]
    un_new = un_new.rename(columns={"Estimate!!Unemployment rate!!Population 16 years and over": "unemployment"})
    edu_new = zip_split(education)[["zip",
                                    "Estimate!!Percent!!Population 25 years and over!!Percent high school graduate or higher",
                                    "Estimate!!Percent!!Population 25 years and over!!Percent bachelor's degree or higher"]]
    edu_new = edu_new.rename(columns={
        "Estimate!!Percent!!Population 25 years and over!!Percent high school graduate or higher": "High school or higher",
        "Estimate!!Percent!!Population 25 years and over!!Percent bachelor's degree or higher": "Bachelor or higher"})
    pov_new = zip_split(poverty)[["zip",
                                  "Estimate!!Percent below poverty level!!Population for whom poverty status is determined"]]
    pov_new = pov_new.rename(
        columns={"Estimate!!Percent below poverty level!!Population for whom poverty status is determined": "poverty"})

    # Construct new dataset for correlation analysis
    db = un_new.merge(edu_new, on="zip").merge(pov_new, on="zip").merge(gv_new, on="zip", how="right")
    db1 = db[["zip", "city", "cases", "n_killed", "n_injured", "population", "rate", "High school or higher",
              "Bachelor or higher", "poverty", "unemployment"]]

    # Handle NAN values and show the descriptive statistics
    db2 = db1.dropna()
    db3 = count_inf_none(db2, db2.columns[2:])
    print("The descriptive statistics of variables:")
    print(db3.describe())
    # Check the histgrams of all variables
    for column in ["rate", "High school or higher", "Bachelor or higher", "unemployment", "poverty"]:
        fig, ax = plt.subplots(figsize=(20, 5))
        sns.histplot(db3[column])
        plt.show()

    # Due to the skewness of rate, use MAD to clean outliers
    outlier_mad = mad_method(db3, 'rate')
    print(outlier_mad)
    db5 = db3.drop(outlier_mad, errors="ignore").reset_index()
    # Check the descriptive statistics again, but find the effect of MAD is not so obvious
    print(db5.describe())
    print("Check the descriptive statistics again, but find the effect of MAD is not so obvious, "
          "so use the original samples")
    # Then decide use the original samples

    # Divide rates into 8 levels for further analysis
    # Levels 0-7 are split according to median-3 * std, median - 2 * std, median - std, median, median + std,
    # median+2 * std, and median+3 * std
    median = np.median(np.log10(db3["rate"]))
    std = np.nanstd(np.log10(db3["rate"]))
    bins = [-100, median - 3 * std, median - 2 * std, median - std, median, median + std, median + 2 * std,
            median + 3 * std, 100]
    db3["level"] = pd.cut(np.log10(db3["rate"]), bins=bins, include_lowest=True, labels=[0, 1, 2, 3, 4, 5, 6, 7])
    level_count = db3.groupby(["level"]).count()
    db3["level"] = db3["level"].astype("int")

    """Correlations"""
    # Draw barplots to show the relationships between gun violence levels and other variables
    for column in ["High school or higher", "Bachelor or higher", "unemployment", "poverty"]:
        draw_relation(db3, "level", column)

    # Export variables again
    db3.to_csv("Variables.csv")

    """Multi Linear Regression"""
    db = db3
    # Reformat the variables
    pov = db["poverty"] / 100
    un = db["unemployment"] / 100
    edu = db["High school or higher"] / 100
    edu2 = db["Bachelor or higher"] / 100
    xv1 = pd.DataFrame([edu, pov, un]).T
    xv2 = pd.DataFrame([edu2, pov, un]).T
    xv3 = pd.DataFrame([edu, edu2, pov, un]).T
    yv = pd.DataFrame(np.log10(db["rate"]))

    model1 = linear(xv1, yv)
    model2 = linear(xv2, yv)
    print("Educational level defined by the Bachelor's Degree or higher is more contributed to GV rates than defined "
          "by the high school or higher.")
    dba = xv3.merge(yv, left_index=True, right_index=True)
    # Plot correlation metrix and pairwise metrix
    print(plot_correlation(dba))
    sns.pairplot(dba)

    """Further Analysis"""
    # import other variables age and sex ratio
    age_file = pd.read_csv("age-2017.csv", header=1, low_memory=False)
    age_new = zip_split(age_file)[
        ["zip", "Estimate!!Percent!!Total population!!SELECTED AGE CATEGORIES!!16 years and over",
         "Estimate!!Percent Male!!Total population!!SELECTED AGE CATEGORIES!!16 years and over"]].rename(
        columns={"Estimate!!Percent!!Total population!!SELECTED AGE CATEGORIES!!16 years and over": "age",
                 "Estimate!!Percent Male!!Total population!!SELECTED AGE CATEGORIES!!16 years and over": "sex"})
    # age_new["zip"] = age_new["zip"].astype("float64")
    db5 = db.merge(age_new, on="zip")
    db5["age"] = db5["age"].astype("float64")
    db5["sex"] = db5["sex"].astype("float64")
    describe = db5.describe()
    print(describe)

    age = db5["age"].astype("float64") / 100
    sex = db5["sex"].astype("float64") / 100
    xv = pd.DataFrame([edu2, pov, un, sex, age]).T
    yv = pd.DataFrame(np.log10(db["rate"]))

    dba2 = xv.merge(yv, right_index=True, left_index=True)
    print(plot_correlation(dba2))
    sns.pairplot(dba2)

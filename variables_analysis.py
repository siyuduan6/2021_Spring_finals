import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import time

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

def descriptive(db):
    """
    Do descriptive statistics
    :param db: dataframe containing variables
    :return: the result of descriptive statistics
    >>> file = pd.DataFrame({"a": [35004,35005,35006, 205,446,888, 35006], "b":[5, 2.5,8,6, 5,10,8]})
    >>> descriptive(file)
                      a          b
    count      7.000000   7.000000
    mean   20222.857143   6.357143
    std    18437.967806   2.495234
    min      205.000000   2.500000
    25%      667.000000   5.000000
    50%    35004.000000   6.000000
    75%    35005.500000   8.000000
    max    35006.000000  10.000000
                      a          b
    count      7.000000   7.000000
    mean   20222.857143   6.357143
    std    18437.967806   2.495234
    min      205.000000   2.500000
    25%      667.000000   5.000000
    50%    35004.000000   6.000000
    75%    35005.500000   8.000000
    max    35006.000000  10.000000


    """

    describe = db.describe()
    print(describe)
    return describe


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
        print(name, db[db[name] == "-"][name].count())
        db_new = db_new.drop(db_new[db_new[name] == np.inf].index)
        db_new = db_new.drop(db_new[db_new[name] == "-"].index)
        db_new = db_new.astype({name: "float64"})
        db_new.to_csv("Variables.csv")
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
    #Takes two parameters: dataframe & variable of interest as string
    med = np.median(df[variable_name])
    mad = np.abs(stats.median_absolute_deviation(df[variable_name]))
    threshold = 3
    outlier = []
    for i, v in enumerate(df.loc[:,variable_name]):
        t = (v-med)/mad
        if t > threshold or t < -1*threshold:
            outlier.append(i)
        else:
            continue
    return outlier

def head(db, i, x):
    """
    Extract the top x sorted samples
    :param db: the dataframe for analysis
    :param i: the subset to sort the dataset
    :param x:the xth highest number
    :return: the sorted dataset
    >>> file3 = pd.DataFrame({"a": [35004,35005,35006, 205,446,888, 35006], "b":[5, 2.5,8,6, 5,10,8]})
    >>> print(head(file3, "a", 5))
           a     b
    2  35006   8.0
    6  35006   8.0
    1  35005   2.5
    0  35004   5.0
    5    888  10.0
    """
    sort = db.sort_values(i, ascending = False).head(x)
    return sort

def draw_relation(db, v, i):
    """
    Sort the dataframe by a specific variable and draw the relationship between two variables as a barplot
    :param db: dataframe containing variables
    :param v: x variable
    :param i: y variable
    :return:
    """
    fig, ax = plt.subplots(figsize=(20,5))
    plt.xticks(rotation=30)
    sns.barplot(x=db[v], y=db[i], palette="Blues_d")
    plt.show()

if __name__ == '__main__':

    # Import files
    pop = pd.read_csv("pop-2017.csv", header=1, low_memory=False)
    unemployment = pd.read_csv("unemployment-2017.csv", header=1, low_memory=False)
    education = pd.read_csv("educational_attainment_2017.csv", header=1, low_memory=False)
    poverty = pd.read_csv("poverty-2017.csv", header=1, low_memory=False)
    gv = pd.read_csv("GV_zip_cleaned.csv", header=0)
    gv_new = gv.groupby(["zip"])["incident_id"].count().to_frame()
    gv_new1 = gv.groupby(["zip"])[["n_killed", "n_injured"]].agg({"n_killed": sum, "n_injured":sum })
    gv_new["n_killed"] = gv_new1.iloc[:,0]
    gv_new["n_injured"] = gv_new.iloc[:,1]
    gv_new = gv_new.reset_index()
    # Transform the format of zip code
    gv_new["zip"] = gv_new["zip"].astype("str")
    gv_new = gv_new.rename(columns={"incident_id":"cases"})
    k = 0
    while k < gv_new.shape[0]:
        if len(gv_new.loc[k,"zip"]) == 4:
            gv_new.loc[k,"zip"] = "0"+gv_new.loc[k,"zip"]
        k += 1
    # Extract variables
    gv_new = gv_new.merge(zip_split(pop), on="zip")
    gv_new["population"] = gv_new["Estimate!!Total!!Total population"]
    gv_new["rate"] = gv_new["cases"]/(gv_new["population"]/1000)
    gv["zip"] = gv["zip"].astype("str")
    gv_new["city"] = gv_new.merge(gv, on="zip")["city_or_county"]
    un_new = zip_split(unemployment)[["zip",
                                      "Estimate!!Total!!Population 16 years and over",
                                      "Estimate!!Unemployment rate!!Population 16 years and over"]]
    un_new = un_new.rename(columns= {"Estimate!!Unemployment rate!!Population 16 years and over":"unemployment"})
    edu_new = zip_split(education)[["zip",
                                    "Estimate!!Percent!!Population 25 years and over!!Percent high school graduate or higher",
                                    "Estimate!!Percent!!Population 25 years and over!!Percent bachelor's degree or higher"]]
    edu_new = edu_new.rename(columns = { "Estimate!!Percent!!Population 25 years and over!!Percent high school graduate or higher":"High school or higher",
                                        "Estimate!!Percent!!Population 25 years and over!!Percent bachelor's degree or higher":"Bachelor or higher"})
    pov_new = zip_split(poverty)[["zip",
                                  "Estimate!!Percent below poverty level!!Population for whom poverty status is determined"]]
    pov_new = pov_new.rename(columns = {"Estimate!!Percent below poverty level!!Population for whom poverty status is determined":"poverty"})

    # Construct new dataset for correlation analysis
    db = un_new.merge(edu_new, on="zip").merge(pov_new, on="zip").merge(gv_new, on="zip", how="right")
    db1 = db[["zip","city","cases", "n_killed","n_injured","population","rate","High school or higher", "Bachelor or higher", "poverty","unemployment"]]
    # Show the descriptive statistics
    descriptive(db1)

    # Handle NAN values and see the result again
    db2 = db1.dropna()
    db3 = count_inf_none(db2, db2.columns[2:])
    descriptive(db3)
   # Check the histgrams of all variables
    #sns.histplot(db3["rate"])
    #time.sleep(10)
    #sns.histplot(np.log10(db3["rate"]))
    #time.sleep(10)
    #sns.histplot(db3["High school or higher"])
    #time.sleep(10)
    #sns.histplot(db3["Bachelor or higher"])
    #time.sleep(10)
    #sns.histplot(db3["unemployment"])
    #time.sleep(10)
    #sns.histplot(db3["education"])

    # Due to the skewness of rate, use MAD to clean outliers
    outlier_mad = mad_method(db3, 'rate')
    print(outlier_mad)
    db5 = db3.drop(outlier_mad, errors="ignore").reset_index()
    # Check the descriptive statistics again, but find the effect of MAD is not so obvious
    # Then decide use the original samples
    descriptive(db5)

    # Divide rates into 8 levels for further analysis
    median = np.median(np.log10(db3["rate"]))
    std = np.nanstd(np.log10(db3["rate"]))
    bins = [-100, median-3 * std, median - 2 * std, median - std, median, median + std, median+2 * std, median+3 * std, 100]
    db3["level"] = pd.cut(np.log10(db3["rate"]), bins=bins, include_lowest=True, labels=[0,1,2,3,4,5,6,7])
    describe_new = db3.describe()
    level_count = db3.groupby(["level"]).count()
    db3["level"] = db3["level"].astype("int")

    for column in db3.columns:
        draw_relation(db3, "level", column)

    # export db4
    db3.to_csv("Variables.csv")



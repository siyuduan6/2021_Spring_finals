import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

def zip_split(file):
    """
    Split the zip code from address.
    :param file: file containing variables
    :return: file with new column:zip
    """
    file["zip"] = file["Geographic Area Name"].str.split(" ", expand=True)[1].astype("str")
    return file


def count_inf_none(db, column_names):
    """
    Find whether there are NAN values in each column and handle them.
    :param db: dataset you what to deal with
    :param column_names: the columns you want to deal with
    :return: a dataset without NAN values
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
    Do MAD
    :param df:
    :param variable_name:
    :return:
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

def draw_hist(db):
    sns.hist(db["rate"])
    sns.hist(db["High school or higher"])
    sns.hist(db["Bachelor or higher"])
    sns.hist(db["unemployment"])
    sns.hist(db["education"])

def head(db, i, x):
    """
    Extract the top x sorted samples
    :param db:
    :param i:
    :param x:
    :return:
    """
    sort = db.sort_values(i, ascending = False).head(x)
    return sort

def draw_head20(v,i):
    """
    Draw the top 20
    :param v:
    :param i:
    :return:
    """
    head_city = head(db3, i, 20)
    fig, ax = plt.subplots(figsize=(20,5))
    plt.xticks(rotation=30)
    sns.barplot(x=head_city[v], y=head_city[i], palette="Blues_d")

def draw_head(db, v, i):
    """
    Draw the top i
    :param db:
    :param v:
    :param i:
    :return:
    """

    head_city = head(db, i, db.shape[0])
    fig, ax = plt.subplots(figsize=(20,5))
    plt.xticks(rotation=30)
    sns.barplot(x=head_city[v], y=head_city[i], palette="Blues_d")

if __name__ == '__main__':

    # Import files
    pop = pd.read_csv("pop-2017.csv", header=1, low_memory=False)
    unemployment = pd.read_csv("unemployment-2017.csv", header=1, low_memory=False)
    education = pd.read_csv("educational_attainment_2017.csv", header=1, low_memory=False)
    poverty = pd.read_csv("poverty-2017.csv", header=1, low_memory=False)
    gv = pd.read_csv("IS597_zip_cleaned.csv", header=0)
    age_file = pd.read_csv("age-2017.csv", header=1, low_memory=False)

    gv_new = gv.groupby(["zip"])["incident_id"].count().to_frame()
    gv_new1 = gv.groupby(["zip"])[["n_killed", "n_injured"]].agg({"n_killed": sum, "n_injured":sum })
    gv_new["n_killed"] = gv_new1.iloc[:,0]
    gv_new["n_injured"] = gv_new.iloc[:,1]
    gv_new = gv_new.reset_index()
    # Transform the format of zip code
    gv_new["zip"] = gv_new["zip"].astype("str")
    gv_new = gv_new.rename(columns = {"incident_id":"cases"})
    k = 0
    while k < gv_new.shape[0]:
        if len(gv_new.loc[k,"zip"]) == 4:
            gv_new.loc[k,"zip"] = "0"+gv_new.loc[k,"zip"]
        k += 1
    # Extract variables
    gv_new = gv_new.merge(zip_split(pop), on="zip")[["zip","Estimate!!Total!!Total population"]]
    gv_new["population"] = gv_new.merge(zip_split(pop), on="zip", how ="left")["Estimate!!Total!!Total population"]
    gv_new["rate"] = gv_new["cases"]/(gv_new["population"]/1000)
    gv["zip"] =  gv["zip"].astype("str")
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

    age_new = zip_split(age_file)[["zip","Estimate!!Percent!!Total population!!SELECTED AGE CATEGORIES!!16 years and over",
                                   "Estimate!!Percent Male!!Total population!!SELECTED AGE CATEGORIES!!16 years and over"]].\
        rename(columns= {"Estimate!!Percent!!Total population!!SELECTED AGE CATEGORIES!!16 years and over":"age",
    "Estimate!!Percent Male!!Total population!!SELECTED AGE CATEGORIES!!16 years and over":"sex"})

    # Construct new dataset for correlation analysis
    db = un_new.merge(edu_new, on="zip").merge(pov_new, on="zip").merge(gv_new, on="zip", how="right")
    db1 = db[["zip","city","cases", "n_killed","n_injured","population","rate","High school or higher", "Bachelor or higher", "poverty","unemployment"]]
    db2 = db1.dropna()
    # Show the descriptive statistics
    describe = db.describe()
    print(describe)
    # Handle NAN values and see the result again
    db3 = count_inf_none(db2, db2.columns[2:])
    describe1 = db3.describe()
    print(describe1)

    # Check the histgrams of all variables
    sns.histplot(db3["rate"])
    sns.barplot(np.log10(db3["rate"]))
    sns.histplot(np.log10(db3["rate"]))
    draw_hist(db3)

    # Due to the skewness of rate, use MAD to clean outliers
    outlier_mad = mad_method(db3, 'rate')
    print(outlier_mad)
    db4 = db3.drop(outlier_mad, errors="ignore").reset_index()

    # Divide rates into 8 levels for further analysis
    median = np.median(np.log10(db4["rate"]))
    median = np.median(np.log10(db4["rate"]))
    bins = [-100, median-3*std, median-2*std, median-std, median, median+std, median+2*std, median +3*std, 100]
    db4["level"] = pd.cut(np.log10(db4["rate"]),
                          bins=bins, include_lowest=True, labels=[0,1,2,3,4,5,6,7])
    describe_new = db4.describe()
    level_count = db4.groupby(["level"]).count()

    for column in db4.columns:
        draw_head(db4, "level", column)

    # export db4
    db4.to_csv("Variables_for_analysis.csv")



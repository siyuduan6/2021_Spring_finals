import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pop = pd.read_csv("pop-2017.csv", header=1, low_memory=False)
unemployment = pd.read_csv("unemployment-2017.csv", header=1, low_memory=False)
education = pd.read_csv("educational_attainment_2017.csv", header=1, low_memory=False)
poverty = pd.read_csv("poverty-2017.csv", header=1, low_memory=False)
gv = pd.read_csv("IS597_zip_cleaned.csv", header=0)
# print(gv[gv["zip"]== ""])
# gv_na = gv[["zip","address"]].isna().sum().reset_index().rename(columns={0:"missing"})

# print(gv_na)
# gv = gv.dropna(axis=0, subset=["zip"])
gv_new = pd.DataFrame()
gv_new[["cases", "n_killed","n_injured"]] = gv[["incident_id","n_killed","n_injured"]].groupby("zip").agg(["count","sum","sum"])
gv_new["population"] = gv_new.merge(pop[["zip","Estimate!!Total!!Total population"]], on="zip")
gv_new["rate"] = gv_new["cases"]/(gv_new["population"]/1000)

def zip_split(file):
    file["zip"] = file["Geographic Area Name"].str.split(" ", expand=True)[1]
    for i in file.columns:
        print(i)
    return file

un_new = zip_split(unemployment)[["zip",
                                  "Estimate!!Total!!Population 16 years and over",
                                  "Estimate!!Unemployment rate!!Population 16 years and over"]]
edu_new = zip_split(education)[["zip",
                                "Estimate!!Total!!Population 25 years and over!!High school graduate (includes equivalency)",
                                "Estimate!!Total!!Population 25 years and over!!Some college, no degree",
                                "Estimate!!Total!!Population 25 years and over!!Associate's degree",
                                "Estimate!!Total!!Population 25 years and over!!Bachelor's degree",
                                "Estimate!!Total!!Population 25 years and over!!Graduate or professional degree",
                                "Estimate!!Total!!Population 25 years and over"]]
ed_new = pd.DataFrame(columns={"zip": edu_new["zip"]})
ed_new["High school or higher"] = (edu_new.iloc[:, 1] + edu_new.iloc[:, 2] + edu_new.iloc[:, 3] + edu_new.iloc[:, 4]
                                  + edu_new.iloc[:, 5])/edu_new["Estimate!!Total!!Population 25 years and over"]
ed_new["Bachelor or higher"] = (edu_new.iloc[:, 4]+edu_new.iloc[:, 5])/edu_new["Estimate!!Total!!Population 25 years and over"]
pov_new = zip_split(poverty)[["zip",
                              "Estimate!!Percent below poverty level!!Population for whom poverty status is determined"]]

db = un_new.merge(ed_new, on="zip").merge(pov_new, on="zip").merge(gv_new, on="zip")

describe = db.describe()
print(describe)
# Educational level

ed_sort1 = ed_new.sort_values("High school or higher").head(50)
ed_sort2 = ed_new.sort_values("Bachelor or higher").head(50)
pov_sort = pov_new.sort_values("Estimate!!Percent below poverty level!!"
                               "Population for whom poverty status is determined").rename(columns="Poverty").head(50)
un_sort = un_new.sort_values("Estimate!!Unemployment rate!!Population 16 years and over").rename(columns="Unemployment").head(50)
db_sort = db.sort_values("cases").head(50)
db_sort2 = db.sort_values("rate").head(50)
# High School or higher
sns.barplot(x=ed_sort1["zip"], y=ed_sort1["High school or higher"], palette="Blues_d")
# Bachelors or Higher
sns.barplot(x=ed_sort2["zip"], y=ed_sort2["Bachelor or higher"], palette="Blues_d")
# Poverty
sns.barplot(x=pov_sort["zip"], y=pov_sort["Poverty"], palette="Blues_d")
# Unemployment
sns.barplot(x=un_sort["zip"], y=un_sort["Unemployment"], palette="Blues_d")
# Top number of cases
sns.barplot(x=db_sort["zip"], y=[0,10,20,30,40,50,60,70,80,90,100],
            hue=db_sort[["High school or higher",
                         "Bachelor or higher",
                         "Estimate!!Unemployment rate!!Population 16 years and over"
                         "Estimate!!Percent below poverty level!!Population for whom poverty status is determined"
                         ]],
            palette="light:b")

# Histogram
# Cases:
sns.hist(db["rates"])
sns.hist(db["High school or higher"])
sns.hist(db["Bachelor or higher"])
sns.hist(db["Estimate!!Unemployment rate!!Population 16 years and over"])
sns.hist(db["Estimate!!Percent below poverty level!!Population for whom poverty status is determined"])



# The distribution

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.kernel_ridge import KernelRidge
from statsmodels.api import OLS
from scipy import stats


# GV_rate = the number of GV cases / the population of the region /10000)
# the number of case per 1000 people
# edu = the percentage of people over 25 years old with high school or higher degree
# pov = the percentage of people who are under poverty level
# un = the percentage of people over 16 years old who are unemployed

def linear(X, y):
    lr = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    model = lr.fit(X_train, y_train)
    summery = model.summery()
    print(summery)
    y_pred = model.predict(X_test)
    plot_model(X_train, X_test, y_train, y_pred)
    score = model.score(X_test, y_test)
    return model

def plot_model(X_train, X_test, y_train,pred):
    sns.relpolt(X_train, y_train, edgecolors=(0, 0, 0))
    sns.relplot(X_test, pred, lw=3, colors = "")
    plt.show()

def plot_correlation(x):
    corr = x.corr(method="spearman")
    plt.matshow(corr)


# Multi Linear Regression
db =
un = db["Unemployment rate"] / 100
pov = db["Poverty rate"] / 100
edu = db["Educational attainment rate"] / 100
xv = pd.DataFrame([edu, pov, un])
yv = pd.DataFrame(db["GV rate"])
linear(xv, yv)
lr = LinearRegression()

scores = cross_val_score(lr, X=pd.DataFrame([edu, pov, un]), y=db["GV rate"], cv=10)
print(scores)
#Linear Regression


#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn import linear_model

data = pd.read_csv("scripts/mtcars.csv")

data.head()

x_train = data.iloc[:,2:]
y_train = data['mpg']

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

x_train.columns.values

col_imp = ['cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']

regr.predict(x_train.iloc[30:])

dict_values = {'name': 'Volvo 142E', 'cyl': 4, 'disp': 121.0, 'hp': 109, 'drat':4.11, 'wt': 2.78, 'qsec': 18.6, 'vs': 1, 'am': 1, 'gear': 4, 'carb':2}

def predict(dict_values, col_imp=col_imp, clf=regr):
    x = np.array([float(dict_values[col]) for col in col_imp])
    x = x.reshape(1,-1)
    y_pred = clf.predict(x)[0]
    name  = dict_values.get('name')
    return {'MPG prediction': {name: y_pred}}
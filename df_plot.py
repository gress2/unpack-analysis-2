import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# apply a filter (essentially a dictionary mapping keys to values) to the dataframe
def filter_df(filter, df):
    for key in filter:
        df = df[(df[key] == filter[key])]
    return df

# get the set of unique values for each column of the dataframe (minus timing)
def unique_set(df):
    unique = {}
    columns = list(df.columns.values)
    columns.remove('timing')
    for column in columns:
        unique[column] = df[column].unique() 
    return unique

# prints out the type of data you'll get after applying a filter to a dataframe
def get_selection_type(unique, filter_dict):
    print('Querying with this filter will yield data with these dimensions unspecified:\n')
    selection_type = {}
    for key in unique:
        if key not in filter_dict:
            selection_type[key] = unique[key]
    return selection_type
    
# expects an array of timing means and a complementing numeric array from another dimension  
# ASSUMPTION: sorted 
def linearity_test(x_axis, y_axis, plot=False):
    x_axis_2d = np.matrix(x_axis).T
    regr = linear_model.LinearRegression()	 
    regr.fit(x_axis_2d, y_axis)
    print('R2 score: %.8f' % regr.score(x_axis_2d, y_axis))
    print('An R2 of 1 is a perfect fit. Range: (-infty, 1]')
    if plot is False:
        return
    y_pred = regr.predict(x_axis_2d)
    plt.scatter(x_axis, y_axis, color='black')
    plt.plot(x_axis, y_pred, color='blue', linewidth='2')
    plt.show()

# build an x, y set from dataframe. if y_col is 'timing', will use the timing mean
def xy(dataframe, x_col, y_col, sortx=True):
    df = dataframe
    x = []
    y = []
    if sortx is True:
        df = df.sort_values(x_col)
    for idx, row in df.iterrows():
        x.append(row[x_col])
        if y_col is 'timing': # it probably is
            y.append(np.mean(row['timing']))
        else:
            y.append(row[y_col])
    return x, y


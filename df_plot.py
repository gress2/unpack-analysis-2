from collections import OrderedDict
import copy
import itertools
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# apply a filter (essentially a dictionary mapping keys to values) to the dataframe
def filter_df(filter, df):
    for key in filter:
        df = df[(df[key] == filter[key])]
    return df

def multi_filter_df(filters, df):
    subframes = []
    for filter in filters:
        subframe = copy.deepcopy(df)
        for key in filter:
            subframe = subframe[(subframe[key] == filter[key])]
        subframes.append(subframe)
    return pd.concat(subframes)

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
def linearity_test(x_axis, y_axis, show_output=False):
    x_axis_2d = np.matrix(x_axis).T
    regr = linear_model.LinearRegression()	 
    regr.fit(x_axis_2d, y_axis)
    r2 = regr.score(x_axis_2d, y_axis)
    if show_output:
        print('R2 score: %.8f' % r2)
        print('An R2 of 1 is a perfect fit. Range: (-infty, 1]')
        y_pred = regr.predict(x_axis_2d)
        print(y_pred)
        plt.scatter(x_axis, y_axis, color='black')
        plt.plot(x_axis, y_pred, color='blue', linewidth='2')
        plt.show()
    return r2 

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

def multi_xy(df, x_col, y_col, multi, sortx=True):
    if sortx:
        df = df.sort_values(x_col)
    multi_dict = {}
    multi_keys = df[multi].unique()
    for key in multi_keys:
        multi_dict[key] = { 'x': [], 'y': [] }
    for idx, row in df.iterrows():
        x = row[x_col] 
        if y_col is 'timing':
            y = np.mean(row['timing'])
        else:
            y = row[y_col]
        multi_dict[row[multi]]['x'].append(x) 
        multi_dict[row[multi]]['y'].append(y) 
    return multi_dict

def m2_xy(df, x_col, y_col, multi1, multi2, sortx=True):
    ret = dict()
    multi2_keys = df[multi2].unique()
    for key in multi2_keys:
        reduced_df = df[(df[multi2] == key)]
        ret[key] = multi_xy(reduced_df, x_col, y_col, multi1, sortx)
    return ret

def all_filter_dicts(unique_set, let_vary):
    unique_mutable = copy.deepcopy(unique_set)
    filter_dicts = []
    parameter_space = []
    for param in let_vary:
        del unique_mutable[param]
    key_list = list(unique_mutable.keys())
    # define the parameter space
    for key in unique_mutable:
        parameter_space.append(unique_mutable[key])
    combinations = itertools.product(*parameter_space)
    for combination in combinations:
        filter_dict = {}
        for i in range(1, len(combination)):
            filter_dict[key_list[i]] = combination[i]
        filter_dicts.append(filter_dict)
    return filter_dicts

# go through every parameter configuration (keeping all constant except # iterations)
# and find R2 score from linear regression. if R2 > some threshold, print the problem-
# atic configuration. Requires the dataframes "unique_set" and the dataframe itself 
def linearity_test_all(unique_set, df, test_dimension):
    filter_dicts = all_filter_dicts(unique_set, let_vary=[test_dimension])
    for filter_dict in filter_dicts:
        filtered_df = filter_df(filter_dict, df) 
        x, y = xy(filtered_df, x_col=test_dimension, y_col='timing', sortx=True)
        print(linearity_test(x, y))

def experiment_filter_dicts(unique_set, must_haves, let_vary):
    unique_mutable = copy.deepcopy(unique_set) 
    for key in must_haves:
        unique_mutable[key] = [must_haves[key]] 
    return all_filter_dicts(unique_mutable, let_vary)

def make_tuple_types(repeated_types, repeats_ary):
    types = []
    for repeated_type in repeated_types:
        for repeat_num in repeats_ary:
            str_ary = [repeated_type] * repeat_num
            type = 'std::tuple<' + ', '.join(str_ary) + '>'
            types.append(type)
    return types
        
def constrain_types(filter_dicts, allowed_types):
    cleaned_dicts = [] 
    for filter_dict in filter_dicts:
        if filter_dict['type'] in allowed_types:
            cleaned_dicts.append(filter_dict)
    return cleaned_dicts

def get_type_base(type):
    first_open = type.find('<')
    first_comma = type.find(',')
    if first_comma == -1:
        first_comma = type.find('>')
    return type[first_open + 1:first_comma]

def numeric_types(dataframe):
    for idx, row in dataframe.iterrows():
        cur_type = dataframe.loc[idx, 'type']
        dataframe.loc[idx, 'type'] = cur_type.count(',') + 1
    return dataframe

def add_type_bases(dataframe):
    for idx, row in dataframe.iterrows():
        cur_type = dataframe.loc[idx, 'type']
        dataframe.loc[idx, 'type_base'] = get_type_base(cur_type)
    return dataframe

def multi_line_plot(multi_xy):
    for key in multi_xy:
        x = multi_xy[key]['x']
        y = multi_xy[key]['y']
        plt.plot(x, y, label=key)
    plt.legend()
    plt.show()

def multi_bar(multi_xy):
    keys = list(multi_xy.keys())
    bar_width = 0.2
    opacity = 0.8
    offset = 0
    
    x = multi_xy[keys[0]]['x'] 
    inds = np.arange(len(x))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for key in multi_xy:
        yvals = np.array(multi_xy[key]['y'])
        ax.bar(inds+offset, yvals, bar_width, label=key)
        offset += bar_width
    plt.xticks(inds + bar_width, list(x))
    plt.legend()
    plt.show()

def m2_bar(m2_xy, color_ary=['orange', 'blue'], label_rotation=70, label_shift=-.35):
    outer_keys = list(m2_xy.keys()) 
    bar_width = 0.2
    opacity = 0.8
    fig = plt.figure()
    ax = fig.add_subplot(111)
    offset = 0
    sum_inds = 0
    xticks_x = []
    xticks_val = []
    mid_tick_locs = []
    for outer_key in outer_keys:
        outer_key_dict = m2_xy[outer_key]
        inner_keys = list(outer_key_dict.keys())
        x = outer_key_dict[inner_keys[0]]['x']
        inds = np.arange(float(len(x)))
        inds *= bar_width * len(inner_keys)
        inds += sum_inds
        sum_inds = inds[len(inds) - 1] + .5
        inner_idx = 0
        for inner_key in outer_key_dict:
            yvals = np.array(outer_key_dict[inner_key]['y'])
            ax.bar(inds + offset, yvals, bar_width, label=inner_key, color=color_ary[inner_idx]) 
            offset += bar_width
            inner_idx += 1
        new_ticks_x = (inds + offset - (2 * bar_width)).tolist()
        mid_tick_locs.append(new_ticks_x[int((len(new_ticks_x) - 1)/ 2)]) 
        xticks_x += new_ticks_x 
        xticks_val += list(x)
    plt.xticks(xticks_x, xticks_val, rotation=label_rotation)
    locs, labels = plt.xticks()
    locs = list(locs)
    labels = list(labels)
    for loc in mid_tick_locs:
        locs += [loc]
    for ok in outer_keys:
        labels += [ok]
    plt.xticks(locs, labels)
    for tick in ax.get_xticklabels():
        if tick.get_text() in outer_keys:
            tick.set_rotation(0)
            tick.set_y(label_shift)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

def histogram(selection, plot_dim):
    n, bins, patches = plt.hist(selection[plot_dim], 50, normed=1, \
            facecolor='green', alpha=0.75)
    plt.xlabel(plot_dim)
    plt.ylabel('frequency')
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('fivethirtyeight')
import numpy as np
from sklearn.linear_model import LinearRegression
from itertools import cycle, islice
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Colorblind-friendly colors
cbf = [[0, 158 / 255, 115 / 255], [86 / 255, 180 / 255, 233 / 255],
       [213 / 255, 94 / 255, 0], [0, 114 / 255, 178 / 255],
       [204 / 255, 121 / 255, 167 / 255], [230 / 255, 159 / 255, 0]]

A = pd.read_csv('numeric_tables_0210.csv')
A['Total Carbonized Energy'] = A['Actual Emissions'] + A['Averted Emissions']
G = A.groupby('State')  #split data by state


def model_plot_states(model, quants_to_predict, y_title='', main_title=''):
    for key, state in G:
        X = state['Year'].values.reshape(-1, 1)
        y = state[quants_to_predict]
        model.fit(X, y)
        my_colors = list(islice(cycle(cbf), None, len(quants_to_predict)))

        state_graph = state.plot(x='Year',
                                 y=quants_to_predict,
                                 figsize=(12, 8),
                                 color=my_colors)
        state_graph.axvline(x=2009, color='black',
                            linewidth=1)  #graph 2009 line
        plt.legend(loc='best')

        model_data = predicted_df(model, quants_to_predict, state)
        model_data.plot(ax=state_graph,
                        x='Year',
                        y=quants_to_predict,
                        linestyle='dashed',
                        color=my_colors,
                        legend=False,
                        linewidth=1.2)

        #plt.plot(years, reg.predict(array_years), linestyle='dashed') #regression plot

        plt.ylabel(y_title)
        plt.title(main_title + ' in ' + key)
        plt.savefig(str(key) + '_' + y_title + '.png')
        plt.show()


def plot_states(quants_to_predict, y_title='', main_title='', subplots=False):
    if not subplots:
        for key, state in G:
            X = state['Year'].values.reshape(-1, 1)
            y = state[quants_to_predict]
            my_colors = list(islice(cycle(cbf), None, len(quants_to_predict)))

            state_graph = state.plot(x='Year',
                                     y=quants_to_predict,
                                     figsize=(12, 8),
                                     color=my_colors)
            state_graph.axvline(x=2009, color='black',
                                linewidth=1)  #graph 2009 line
            plt.legend(loc='best')

            plt.ylabel(y_title)
            plt.title(main_title + ' in ' + key)
            plt.savefig(str(key) + '_' + y_title + '.png')
            plt.show()
    else:
        fig, axes = plt.subplots(nrows=2, ncols=2)
        i = 0
        plttest = []
        for key, state in G:
            X = state['Year'].values.reshape(-1, 1)
            y = state[quants_to_predict]
            my_colors = list(islice(cycle(cbf), None, len(quants_to_predict)))

            p = state.plot(ax=axes.flat[i],
                           x='Year',
                           y=quants_to_predict,
                           figsize=(8, 8),
                           color=my_colors)
            axes.flat[i].set_title(key)
            plttest.append(p)
            #plt.ylabel(y_title)
            i += 1
        #plt.title(main_title)
        axes[1, 1].legend_.remove()
        axes[0, 1].legend_.remove()
        axes[0, 0].legend_.remove()

        axes[1, 1].xaxis.label.set_visible(False)
        axes[0, 1].xaxis.label.set_visible(False)
        axes[0, 0].xaxis.label.set_visible(False)

        axes[1, 0].set_ylabel(y_title, weight="bold")
        fig.tight_layout()
        plt.savefig((main_title + y_title + '.png').replace(' ', '_'))
        plt.show()


def plot_states4(statekey, quants_array, ylabels,
                 mtitles):  #pass it an array L=[[quants1], [quants2], ...4]
    state = G.get_group(statekey)
    fig, axes = plt.subplots(nrows=2, ncols=2)

    for i in range(4):
        a = axes.flat[i]
        my_colors = list(islice(cycle(cbf), None, len(quants_array[i])))
        state.plot(ax=a,
                   x='Year',
                   y=quants_array[i],
                   color=my_colors,
                   figsize=(12, 12))
        a.set_ylabel(ylabels[i])
        a.set_title(mtitles[i])
    plt.legend(loc='best')
    fig.tight_layout()
    plt.savefig('MAIN_plts' + statekey + '.png')
    plt.show()


def plot_4(statekey, quants_array, ylabels,
           mtitles):  #pass it an array L=[[quants1], [quants2], ...4]
    state = G.get_group(statekey)
    fig, axes = plt.subplots(nrows=2, ncols=2)

    for i in range(4):
        a = axes.flat[i]
        my_colors = list(islice(cycle(cbf), None, len(quants_array[i])))
        state.plot(ax=a,
                   x='Year',
                   y=quants_array[i],
                   color=my_colors,
                   figsize=(12, 12))
        a.set_ylabel(ylabels[i])
        a.set_title(mtitles[i])
    plt.legend(loc='best')
    fig.tight_layout()
    plt.savefig('MAIN_plts' + statekey + '.png')
    plt.show()


def profile_constructor():
    dirty = ['Petroleum', 'Coal', 'Wood and Waste', 'Natural Gas', 'Ethanol']
    clean = ['Geothermal', 'Hydroelectric', 'Solar', 'Wind', 'Nuclear']
    tce = ['Actual Emissions', 'Averted Emissions', 'Total Carbonized Energy']
    g = ['Green Score']
    L = [dirty, clean, tce, g]
    ylabels = ['lbs CO2', 'lbs CO2', 'lbs CO2', '']
    mtitles = [
        'Actual Emissions', 'Averted Emissions', 'Total Carbonized Energy',
        'Green Score'
    ]
    for state in ['AZ', 'CA', 'NM', 'TX']:
        print(state)
        plot_4(state, L, ylabels, mtitles)


#plot_states(['Averted Emissions'], subplots=True)


def plot_both(model=LinearRegression()):
    all_energy = [
        'Petroleum', 'Coal', 'Wood and Waste', 'Natural Gas', 'Ethanol',
        'Geothermal', 'Hydroelectric', 'Solar', 'Wind', 'Nuclear'
    ]
    model_plot_states(model, all_energy, 'CO2 Emissions (lbs)',
                      'Linear Regression on Emissions')


def plot_cln(model=LinearRegression()):
    clean = ['Geothermal', 'Hydroelectric', 'Solar', 'Wind', 'Nuclear']
    model_plot_states(model, clean, 'Averted CO2 Emissions (lbs)',
                      'Linear Regression on Clean Energy')


def plot_drt(model=LinearRegression()):
    dirty = ['Petroleum', 'Coal', 'Wood and Waste', 'Natural Gas', 'Ethanol']
    model_plot_states(model, dirty, 'Actual CO2 Emissions (lbs)',
                      'Linear Regression on Dirty Energy')


def ems_plot():
    A['AE Emissions'] = A['Actual Emissions'] + A['Averted Emissions']
    plot_states(['AE Emissions', 'Actual Emissions', 'Averted Emissions'],
                'CO2 Emissions (lbs)', 'Energy usage normalized by emissions')


def plot_cln_nomodel(main_title=''):
    clean = ['Geothermal', 'Hydroelectric', 'Solar', 'Wind', 'Nuclear']
    plot_states(clean,
                subplots=True,
                main_title=main_title,
                y_title='Averted emissions (lbs CO2)')


def plot_memo_graph(main_title=''):
    stuff = ['Green Score']
    plot_states(stuff,
                subplots=True,
                main_title=main_title,
                y_title='Green Score')


def predicted_df(model,
                 quants_to_predict,
                 df,
                 last=2051):  #expects a df like all AZ data
    array_years = np.arange(1960, last).reshape(-1, 1)
    q = pd.DataFrame(array_years, columns=['Year'])
    X = q['Year']  #train data to 50
    X = X.values.reshape(-1, 1)

    model.fit(X[:50], df[quants_to_predict])  #fit on only first 50 samples
    data = model.predict(X)
    idxs = q.index
    try:
        a, b = data.shape
        for i, col in enumerate(data.T):
            print(q.index)
            print(type(col))
            name = quants_to_predict[i]
            print(name)
            q[name] = pd.DataFrame(col, index=idxs)
        return q
    except ValueError:
        q[quants_to_predict[0]] = pd.DataFrame(data, index=idxs)
        return q


def rfrtest():
    for key, A in G:
        X = A['Year'].values.reshape(-1, 1)
        Y = A['Green Score'].values.reshape(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(X,
                                                            Y,
                                                            random_state=777,
                                                            shuffle=False,
                                                            test_size=.2)
        rfr = RandomForestRegressor(n_estimators=50)
        reg = LinearRegression()
        rfr.fit(x_train, y_train)
        reg.fit(x_train, y_train)
        print('State: {}. RFR Score: {}. Reg Score: {}'.format(
            key, rfr.score(x_test, y_test), reg.score(x_test, y_test)))

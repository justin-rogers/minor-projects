import pandas as pd
from os import listdir
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

"""
    A=pd.read_csv('./time_for_france.csv', header=None)
    B=pd.read_csv('./values_for_france.csv', header=None)
    
    times=A.values.tolist()[0]
    x_changes, y_changes=B.values.tolist()
    z_changes=[1-x_changes[n]-y_changes[n] for n in range(len(times))]
     
    plt.plot(times, x_changes, linewidth=1.5)
    plt.plot(times, y_changes, linewidth=1.5)
    plt.plot(times, z_changes, linewidth=1.5)
    plt.title('continuous')
    plt.legend(['French', 'English', 'Bilingual'], loc='best')
    plt.show()
"""

def graph_country(country, language):
    A=pd.read_csv('./time_for_france.csv', header=None)
    B=pd.read_csv('./'+country+'.csv', header=None)
    
    times=A.values.tolist()[0]
    x_changes, y_changes=B.values.tolist()
    z_changes=[1-x_changes[n]-y_changes[n] for n in range(len(times))]
     
    plt.plot(times, x_changes, linewidth=1.5)
    plt.plot(times, y_changes, linewidth=1.5)
    plt.plot(times, z_changes, linewidth=1.5)
    plt.title(country)
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction of population')
    plt.legend([language.capitalize(), 'English', 'Bilingual'], loc='best')
    plt.tight_layout()
    plt.savefig(country+'_diffeq_model.png')
    plt.show()
    
def graph_canada():
    A=pd.read_csv('./time_for_france.csv', header=None)
    B=pd.read_csv('./old_Canada.csv', header=None)
    
    times=A.values.tolist()[0]
    x_changes, y_changes=B.values.tolist()
    z_changes=[1-x_changes[n]-y_changes[n] for n in range(len(times))]
     
    plt.plot(times, x_changes, linewidth=1.5)
    plt.plot(times, y_changes, linewidth=1.5)
    plt.plot(times, z_changes, linewidth=1.5)
    plt.title('Unscaled model: Canada')
    plt.xlabel('Time (1 unit is 45 years)')
    plt.ylabel('Fraction of population')
    plt.legend(['French', 'English', 'Bilingual'], loc=1)
    plt.tight_layout()
    plt.savefig('old_Canada_diffeq_model.png')
    plt.show()

"""
graph_country('UK', 'Hindi')

csvnames=[name[:-4] for name in listdir('.') if name[-4:]=='.csv'][:10]
csvlangs=['French', 'Portuguese', 'French', 'German', 'Italian', 'Japanese', 'Korean',
          'French', 'Russian', 'Spanish']
for name,lang in zip(csvnames,csvlangs):
    graph_country(name,lang)
"""
graph_canada()

import numpy as np
from numpy.random import binomial, multinomial
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
 
"""
The paper uses an agent-based model where they actually keep track of many
individual agents. But they don't actually make full use of this, since they
assume each agent is connected to each other-- so there is no way to distinguish
between X Speaker 1 and X Speaker 2.
 
Because of this, we can save an enormous amount of computational power by not
tracking every single individual-- instead, we track the number of speakers.
 
For an individual, transition probabilities are given by Bernoulli RVs.
So we are interested in a sum of Bernoulli RVs: which is a binomial distribution (or a multinomial distribution)
 
For n trials and probability p, we draw a sample from binomial(n,p)
"""
 
 
class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
 
class Simulator(Bunch):
    def __init__(self, adict):
        Bunch.__init__(self, adict)
        self.a=1.31
        self.z=1-self.x-self.y
        self.pop_xyz=[round(self.x*self.n), round(self.y*self.n), round(self.z*self.n)]
        self.startcfg=[self.n, self.x, self.y, self.z, self.pop_xyz] #in case we want to restore these
    #self.x, y, z are reserved for percentages.
    #self.pop_zyx[0]=integer x population, etc.
    def reset(self):
        self.n, self.x, self.y, self.z, self.pop_xyz=self.startcfg
 
    #In each step, we independently calculate deaths in each population.
    def iterate(self, years):
        x_history=[self.x]
        y_history=[self.y]
        z_history=[self.z]
        for _ in range(years): #run the given amt of years
            a,b,c=self.step(0), self.step(1), self.step(2)
            #each of these is a 3-element list: [x_i, y_i, z_i]. we add them up:
            new_pops=[sum(x) for x in zip(a,b,c)]
           
            self.n=sum(new_pops)
            #supports changing population
           
            self.pop_xyz=new_pops
            self.x=new_pops[0]/self.n
            self.y=new_pops[1]/self.n
            self.z=new_pops[2]/self.n
           
            x_history.append(self.x)
            y_history.append(self.y)
            z_history.append(self.z)
        #print('After {} years of simulation, our initial percentages of {}, {}, and {} have become {}, {}, and {}.'.format(years, self.startcfg[1], self.startcfg[2], self.startcfg[3], self.x,    self.y, self.z))
        print('x,y,z percentages: {}, {}, {}'.format(self.x, self.y, self.z))
        return x_history, y_history, z_history
        #return self.x, self.y, self.z
 
    def step(self, lang): #1 discrete step of the simulation, input 0/1/2 for x/y/z language
        lang_pop = self.pop_xyz[lang]
        deaths = binomial(lang_pop, self.mu)
        lives = lang_pop-deaths
        v_delta = self.vstep(deaths, lang)
        h_delta = self.hstep(lives, lang)
        pop_change=[sum(x) for x in zip(v_delta,h_delta)]
        return pop_change
   
    def vstep(self, n, lang):
        #input: deaths and 0/1/2 for x/y/z
        #output: a list [a,b,c] of the post-step populations
        if lang==0: #x case: P(X-->X)=1
            return [n,0,0]
        if lang==1: #y
            return [0,n,0]
        if lang==2: #z
            ###Need to use multinomial distribution since there's 3 options.
            px=self.czx*self.sx*self.x**self.a
            py=self.czy*self.sy*self.y**self.a
            pz=1-self.x-self.y
            return list(multinomial(n, [px,py,pz]))
       
    def hstep(self, n, lang):
        #similar specifications as above
        if lang==0:
            pz=self.cxz*self.sy*self.y**self.a
            #px=1-pz
            z=binomial(n,pz)
            return [n-z, 0, z]
        if lang==1:
            pz=self.cyz*self.sx*self.x**self.a
            #py=1-pz
            z=binomial(n,pz)
            return [0, n-z, z]
        if lang==2:
            return [0,0,n]
 
"""
This method might be extended by using birth rate and death rate separately--
and, potentially, using {im, e}migration
"""
 
 
germany_data=({'n':82239678, 'x':0.68070, 'y':0.058300, 'mu':0.011300, 'sx':0.22359,
               'sy':0.77641, 'cxz':0.77867, 'cyz':0.37100, 'czx':0.22133, 'czy':0.62900})
 
canada_data=({'mu':0.0075000, 'x':0.69633, 'y':0.12148, 'sx':0.99080,
             'sy':0.0091999, 'cxz':0.31735, 'cyz':0.77867, 'czx':0.68265,
             'czy':0.22133, 'n':36290000})
"""
germany_data=({'n':822, 'x':0.68070, 'y':0.058300, 'mu':0.011300, 'sx':0.22359,
               'sy':0.77641, 'cxz':0.77867, 'cyz':0.37100, 'czx':0.22133, 'czy':0.62900})
"""
   
test=Simulator(canada_data)
 
 
for _ in range(1):
    x,y,z=test.iterate(50)
    test.reset()
 
 
for var in [x,y,z]:
    plt.plot(range(len(var)), var, linewidth=1.5)
plt.legend(['French', 'English', 'Bilingual'], loc=1)
plt.title('Canada (discrete model)')
plt.xlabel('Time (1 unit is 45 years)')
plt.ylabel('Fraction of Population')
plt.tight_layout()
plt.savefig('discrete_canada_2250years')
plt.show()
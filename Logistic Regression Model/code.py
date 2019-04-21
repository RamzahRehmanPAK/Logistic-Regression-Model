import pandas as pd
import matplotlib.pyplot as plt
from decimal import *
import numpy
from numpy import *

#Hafiza Ramzah Rehman 14-4342
#Sana Fatima 14-4079
def yForDecisionBoundary(dataList,theetas):

    Y=[]
    value=0.0
    features = [1.0]

    for row in dataList:
        for i in range(len(row)-1):
            features.append(row[i])

        features=regularized_features(features)

        for x in range(len(features)):
            if x!=2:
                value-=(features[x]*theetas[x])

        value=(value+0.1)/theetas[2]
        Y.append(value)
        features = [1.0]
    return Y


def getMean(df):
    means=[1.0001]
    df = df.iloc[:, :-1]

    for column in df:
        means.append(df[column].mean())

    return means
#############################
def getStd(df):

    stds = [0.0001]
    df = df.iloc[:, :-1]

    for column in df:
        stds.append(df[column].std())

    return stds
#############################

def linearHypothesis(features,theetas,means,stds):
    returnValue=0.0
    for f,t,m,s in zip(features,theetas,means,stds):
        returnValue+=((f-m)/s)*t
    return returnValue

##############################


def nonlinearHypothesis(features,theetas,means,stds):
    returnValue=0.0
    for f,t,m,s in zip(features,theetas,means,stds):
        returnValue+=((f-m)/s)*t
    return returnValue

##############################

def sigmoidFunction(hypothesis):

    #print(1/(1+ ((2.718281)**(-hypothesis))))
    return 1/(1+ ((2.718281)**(-hypothesis)))

##############################
def get_theetas_list(d):
    no_of_theetas = max(len(d[x]) for x in range(len(d)))
    theetas = []
    for i in range(no_of_theetas):
        theetas.append(0.0)
    return theetas
##########################################
def cost( dataList, theetas,means,stds):
    c=0.0

    features = [1.0]

    for row in dataList:
        for i in range(len(row)-1):
            features.append(row[i])

        #print(sigmoidFunction(linearHypothesis(features,theetas,means,stds)))
        c+= ( ( (-row[len(row)-1]) * math.log(sigmoidFunction(linearHypothesis(features,theetas,means,stds)) ))
              -( ( 1-row[len(row)-1] ) * math.log((1-(sigmoidFunction(linearHypothesis(features,theetas,means,stds)))+(10**(-1000))))));
        features = [1.0]

    c=c/(len(dataList))
    return c
##########################################

def derivative_of_cost( dataList, theetas, theeta_index,means,stds):
    c=0.0
    features = [1.0]

    for row in dataList:
        for i in range(len(row)-1):
            features.append(row[i])

        c += (sigmoidFunction(linearHypothesis(features, theetas,means,stds)) - row[len(row) - 1]) * ((features[theeta_index]-means[theeta_index])/stds[theeta_index])
        features = [1.0]

    c=c/(len(dataList))
    return c
##########################################


def linearRegression(d,means,stds,iterations,alpha):

    theetas = get_theetas_list(d)
    tempTheetas=[ 0.0 for x in theetas]
    prev_cost=10**50
    current_cost=10**50

    #while (prev_cost >= current_cost):
    for i in range(iterations):
        for j in range(len(theetas)):
            tempTheetas[j] = theetas[j] - (alpha * derivative_of_cost(d, theetas, j,means,stds))
        theetas = tempTheetas[:]
        # print(theetas)
        prev_cost = current_cost
        current_cost = cost(d, theetas,means,stds)
        print(current_cost)

    return theetas
########################################

def costForRegularized( dataList, theetas,means,stds,lambdaa):
    c=0.0

    features = [1.0]

    for row in dataList:
        for i in range(len(row)-1):
            features.append(row[i])

        features=regularized_features(features)
        #print(sigmoidFunction(linearHypothesis(features,theetas,means,stds)))
        c+= ( ( (-row[len(row)-1]) * math.log(sigmoidFunction(linearHypothesis(features,theetas,means,stds)) ))
              -( ( 1-row[len(row)-1] ) * math.log((1-(sigmoidFunction(linearHypothesis(features,theetas,means,stds)))+(10**(-1000))))));
        features = [1.0]

    c=c/(len(dataList))

    sumSquareTheetas=0.0
    for k in theetas:
        sumSquareTheetas+=k**2
    sumSquareTheetas=(sumSquareTheetas*lambdaa)/(2*len(dataList))
    c=c+sumSquareTheetas
    return c
##########################################

def derivative_of_costForRegularized( dataList, theetas, theeta_index,means,stds,lambdaa):
    c=0.0
    features = [1.0]

    for row in dataList:
        for i in range(len(row)-1):
            features.append(row[i])

        features=regularized_features(features)
        c += (sigmoidFunction(linearHypothesis(features, theetas,means,stds)) - row[len(row) - 1]) * ((features[theeta_index]-means[theeta_index])/stds[theeta_index])
        features = [1.0]

    c=c/(len(dataList))
    return c
##########################################

def linearRegressionForRegularized(d,means,stds,iterations,alpha,lambdaa):

    #theetas = get_theetas_list(d)
    theetas=[0.0 for x in range(28)]
    tempTheetas=[ 0.0 for x in theetas]
    prev_cost=10**50
    current_cost=10**50

    #while (prev_cost >= current_cost):
    for i in range(iterations):
        for j in range(len(theetas)):
            tempTheetas[j] = (theetas[j]*(1-((alpha*lambdaa)/len(d)))) - (alpha * derivative_of_costForRegularized(d, theetas, j,means,stds,lambdaa))
        theetas = tempTheetas[:]
        # print(theetas)
        prev_cost = current_cost
        current_cost = costForRegularized(d, theetas,means,stds,lambdaa)
        print(current_cost)

    return theetas
########################################

def regularized_features(features):

    def func():
        yield 1
        for i in range(1, 7):
            for j in range(i + 1):
                yield numpy.power(features[1], i - j) * numpy.power(features[2], j)
    new_feature_list= numpy.vstack(func()).tolist()

    list=[]
    for x in new_feature_list:
        for y in x:
            list.append(y)
    return list

def dataPlot(file, name,label1,label2):

    df = pd.read_csv(file, names=name)
    df1 = df[df['result'] == 0]
    df2 = df[df['result'] == 1]
    ax=df1.plot(kind='scatter',x=name[0], y=name[1], color='yellow', marker='o', label=label1)
    df2.plot(ax=ax,kind='scatter',x=name[0], y=name[1], color='black', marker='+', label=label2)


    plt.title('Figure 1: Scatter plot of training data ')
    plt.xlabel('Exam one score')
    plt.ylabel('Exam two score')
    plt.show()

def dataPlotAfterLearning(theetas,means, stds):

    df = pd.read_csv('ex2data1.txt', names=['Exam1_score', 'Exam2_score', 'result'])
    df1 = df[df['result'] == 0]
    df2 = df[df['result'] == 1]
    ax = df1.plot(kind='scatter', x='Exam1_score', y='Exam2_score', color='yellow', marker='o', label='Not admitted')
    df2.plot(ax=ax, kind='scatter', x='Exam1_score', y='Exam2_score', color='black', marker='+', label='Admitted')

    plt.title('Figure 1: Training data with decision boundary ')
    plt.xlabel('Exam one score')
    plt.ylabel('Exam two score')

    x1 = df.Exam1_score.tolist();
    x1=[(a-means[1])/stds[1] for a in x1]
    x1 = numpy.array(x1)

    x0 = [(1 - means[0]) / stds[0] for a in x1]
    x0 = numpy.array(x0)


    plt.plot(df.Exam1_score.tolist(), (((0.1 - (theetas[0]* x0) - (theetas[1] * x1)) / theetas[2]) * stds[2])+means[2] )
    plt.legend(loc='best')
    plt.show()
#######################################


def dataPlotAfterLearningForRegularized(theetas,lambdaa):

    df = pd.read_csv('ex2data2.txt', names=['test1', 'test2', 'result'])
    df1 = df[df['result'] == 0]
    df2 = df[df['result'] == 1]
    ax = df1.plot(kind='scatter', x='test1', y='test2', color='yellow', marker='o', label='y=0')
    df2.plot(ax=ax, kind='scatter', x='test1', y='test2', color='black', marker='+', label='y=1')

    plt.title('Figure 1: Training data with decision boundary lambda='+str(lambdaa))
    plt.xlabel('Test one')
    plt.ylabel('Test two')

    #X = df.test1.tolist();
    #Y=yForDecisionBoundary(df.values.tolist(),theetas)
    #plt.plot(X,Y)

    # Plot Boundary
    u = array(numpy.arange(-1.0, 1.0, 0.01))
    v = array(numpy.arange(-1.0, 1.0, 0.01))
    z = zeros(shape=(len(u), len(v)))

    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = (numpy.array(regularized_features([1,array(u[i]), array(v[j])])).dot(array(theetas)))

    z = z.T
    plt.contour(u, v, z,1, colors='green')
    plt.legend(loc='best')
    plt.show()
#######################################

if __name__ == '__main__':
    print("welcome!")
    alpha=-10
    dataPlot('ex2data1.txt',['Exam1_score','Exam2_score','result'],'Not admitted','Admitted')
    df = pd.read_csv('ex2data1.txt',  names=['Exam1_score','Exam2_score','result'])
    means = getMean(df)
    stds = getStd(df)
    d = df.values.tolist()

   # theetas=linearRegression(d,means,stds,8000,alpha)#0.2035
    theetas=[-1.7090528862898433, 3.992721952200461, 3.7243761258626207]
    dataPlotAfterLearning(theetas,means,stds)
    print(theetas)
    print(cost(d, theetas,means,stds))

    features=[1,45,85]
    print(sigmoidFunction(linearHypothesis(features,theetas,means,stds)))

    ##############################################

    alpha = 6.5
    dataPlot('ex2data2.txt',['test1','test2','result'],'y=0','y=1')

    df = pd.read_csv('ex2data2.txt', names=['test1', 'test2', 'result'])
    means=[0.0 for x in range(28)]
    stds=[1.0 for x in range(28)]

    d = df.values.tolist()

    lambdaa=1
    #theetas=linearRegressionForRegularized(d,means,stds,100,alpha,lambdaa)#0.535160313772012
    theetas=[1.1420099635392726, 0.6012830096959808, 1.1670745023404234, -1.8716704069651857, -0.9151884719694938, -1.2690045560345429, 0.12667435054839643, -0.36875746266468323, -0.3454733690275321, -0.17359025211382959, -1.4237883456843035, -0.04908874106198708, -0.6063007273428731, -0.2692634183841418, -1.1630280836699736, -0.24298235722477216, -0.2072309291525837, -0.04356294425629459, -0.2801341066567282, -0.28696617321923584, -0.4694517434873547, -1.0361500971551976, 0.02884124943893015, -0.29261180142790233, 0.017061930285715066, -0.32886190702177925, -0.13794353355692499, -0.9324229977971693]

    print(theetas)
    print(costForRegularized(d, theetas,means,stds,lambdaa))

    dataPlotAfterLearningForRegularized(theetas,lambdaa)

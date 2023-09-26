import pandas as analytics
import numpy as maths
import matplotlib.pyplot as graph
import os
from math import exp
from linear_regression import linear_regression

class logistic_regression:

    def __init__(self,threshold = 0.5):
        self.threshold = threshold
        
    
    def load_data(self,filename):
        df_data = analytics.read_csv(filename+".csv")
        names = ['x'+str(i+1) for i in range((df_data.shape[1])-1)] + ['y']
        df_data = analytics.read_csv(filename+".csv" , names = names)
        df_data['x0'] = 1
        cols = list(df_data.columns)
        cols = [cols[-1]] + list(cols[:-1])
        df_data = df_data[cols]
        df_data['y'] = df_data['y'].replace(-1,0)
        self.df_data = df_data
        
    def split_dataset(self,validation_perc, training_perc ):
        validation_number = int(len(self.df_data)*validation_perc)
        self.training_perc = training_perc
        
        self.df_data = self.df_data.sample(frac = 1)
        df_validation = self.df_data[:validation_number]
        df_test = self.df_data[validation_number:]
        
        df_validation.iloc[:,1:].to_csv('validation.csv',header=False,index=False)
        
        return df_validation, df_test
    
    def find_weights(self,alphas , k ):
        lr = linear_regression()
        lr.load_data('validation.csv')
        lr.monte_carlo(alphas,k,self.training_perc)
        df_validation_training , df_validation_testing = lr.split_data()
        lr.train(df_validation_training, sgd = True, plot_rmse = False, plot_metrics = True)
        lr.test(df_validation_testing)
        
        self.w_star = lr.w_star
        self.maxima = lr.maxima
        self.minima = lr.minima
        
    def classify(self,df_test):
        for col in range(len(df_test.columns[1:-1])):
            maximum = self.maxima[col]
            minimum = self.minima[col]
            
            df_test.iloc[:,col + 1] = (df_test.iloc[:,col+1] - minimum)/(maximum-minimum)
        df_test['y_hat'] = maths.matmul(df_test.iloc[:,:-1],self.w_star)
        df_test['y_hat'] = df_test['y_hat'].apply(lambda x : 1 if x > self.threshold else 0)
        
        self.create_confusion_matrix(list(df_test['y']),list(df_test['y_hat']))


    def create_confusion_matrix(self,y,y_hat):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(y)):
            if y[i] == y_hat[i] == 1:
                tp = tp + 1
            elif y[i] == y_hat[i] == 0:
                tn = tn + 1
            elif y[i] == 1 and y_hat[i] == 0:
                fn = fn + 1
            else :
                fp = fp + 1

        self.confusion_matrix = {'tp':tp,'tn':tn,'fp':fp,'fn':fn}
        print("Confusion Matrix : ",self.confusion_matrix)
        self.accuracy = self.find_accuracy()
        self.precision = self.find_precision()
        self.sensitivity = self.find_sensitivity()
        self.specificity = self.find_specificity()
        self.fscore = self.find_fscore()


    def find_precision(self,show = True):
        confusion_matrix = self.confusion_matrix
        precision = round((confusion_matrix['tp'])/(confusion_matrix['fp']+confusion_matrix['tp'])*100,2) 
        if show : print("Precision : ",precision,"%")
        return precision

    def find_accuracy(self,show = True):
        confusion_matrix = self.confusion_matrix
        accuracy = round((confusion_matrix['tp']+confusion_matrix['fp'])/(confusion_matrix['tp'] + confusion_matrix['tn'] + confusion_matrix['fp'] + confusion_matrix['fn'])*100,2)
        if show : print("Accuracy : ",accuracy,"%")
        return accuracy

    def find_sensitivity(self,show = True):
        confusion_matrix = self.confusion_matrix
        sensitivity = round((confusion_matrix['tp'])/(confusion_matrix['tp']+confusion_matrix['fn'])*100,2)
        if show : print("Sensivity : ",sensitivity,"%")
        return sensitivity

    def find_specificity(self,show = True):
        confusion_matrix = self.confusion_matrix
        specificity = round((confusion_matrix['tn'])/(confusion_matrix['fp']+confusion_matrix['tn'])*100,2)
        if show : print("Specificity : ",specificity,"%")
        return specificity

    def find_fscore(self,show=True):
        confusion_matrix = self.confusion_matrix
        f_score = round(2/((1/self.find_precision(False))+(1/self.find_sensitivity(False))),2)
        if show : print("F1 Score : ",f_score)
        return f_score        

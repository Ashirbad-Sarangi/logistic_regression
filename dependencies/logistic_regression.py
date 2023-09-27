import pandas as analytics
import numpy as maths
import matplotlib.pyplot as graph
import os
from math import exp
from linear_regression import linear_regression

class logistic_regression:

    def __init__(self,threshold = 0.5):
        self.threshold = threshold
        
        
    def help(self):
        """ Help function to give the details of the Class """
        
        print("""
        
    Variables
    =========
    
        maxima                     # To store the maxima of each attribute of the training set
        minima                     # To store the minima of each attribute of the testing set
        df_data                    # The preprocessed complete dataset
        w_star                     # Optimal Weights selected
        accuracy                   # Optimal Weights selected
        create_confusion_matrix    # Confusion Matrix after the Test Data
        fscore                     # FScore of the Test Data
        precision                  # Precision of the Test Data
        sensitivity                # Sensivity of the Test Data
        specificity                # Specificity of the Test Data
        threshold                  # Threshold for the Classification

    Functions
    =========
        load_data( filename )     
        # Load the data saved with the filename

        split_dataset( validation_perc , training_perc )   
        # Split the dataset into validation set and test set according to the validation_perc. Training part from the validation set is determined by training_perc. So training percentage is training_perc of validation_perc and testing set is 1 - validation_perc. Returns validation set and testing set
        
        find_weights( alphas , number_of_iteration )  
        # Using linear regression to find the weights for setting the hyperplane

        classify( df_test )       
        # Classifying the inputs into the respective class through threshold

        create_confusion_matrix( y , y_hat )       
        # Creating the confusion matrix

        find_precision( show = True )  
        # Finding Precision
        
        find_accuracy( show = True )  
        # Finding Accuracy
        
        find_sensivity( show = True )  
        # Finding Sensivity
        
        find_specificity( show = True )  
        # Finding Specificity
        
        find_fscore( show = True )  
        # Finding Fscore
        

    Sequence of Function Calls 
    ===========================
        # load_data( path )
        # split_dataset( validation_perc , training_perc )
        # find_weights( alphas , number_of_iteration )
        # classify( df_test )

        """)
        
        
    
    def load_data(self,filename):
        """Load the data saved with the filename """
        
        source_folder = "../data/"                                           # defining the actual location of data
        filename = source_folder + filename
        df_data = analytics.read_csv(filename+".csv")                        # read data
        names = ['x'+str(i+1) for i in range((df_data.shape[1])-1)] + ['y']
        df_data = analytics.read_csv(filename+".csv" , names = names)        # read data with names
        
        df_data['x0'] = 1                                                    # append 1 column
        cols = list(df_data.columns)
        cols = [cols[-1]] + list(cols[:-1])
        df_data = df_data[cols]
        
        df_data['y'] = df_data['y'].replace(-1,0)                            # preprocessing
        self.df_data = df_data
        
    def split_dataset(self,validation_perc, training_perc ):
        """ Split the dataset into validation set and test set according to the validation_perc. Training part from the validation set is determined by training_perc. So training percentage is training_perc of validation_perc and testing set is 1 - validation_perc """
        
        validation_number = int(len(self.df_data)*validation_perc)          # Validation Number
        self.training_perc = training_perc
        
        self.df_data = self.df_data.sample(frac = 1)                        # Shuffling data
        df_validation = self.df_data[:validation_number]                    # Distributing data into validation and test
        df_test = self.df_data[validation_number:]                          
        
        df_validation.iloc[:,1:].to_csv('validation.csv',header=False,index=False)
                                                                            # Saving the validation set to take as input in lin reg
        
        return df_validation, df_test
    
    def find_weights(self,alphas , k ):
        """ Using linear regression to find the weights for setting the hyperplane """
        return None
        #---------------------------#
        lr = linear_regression()
        lr.load_data('validation.csv')
        os.remove('validation.csv')
        lr.monte_carlo(alphas,k,self.training_perc)
        df_validation_training , df_validation_testing = lr.split_data()
        lr.train(df_validation_training, sgd = True, plot_rmse = False, plot_metrics = True)
        lr.test(df_validation_testing)
        #--------------------------#
        del self.training_perc
        
        self.w_star = lr.w_star                                            # Store the optimal weights from lin reg
        self.maxima = lr.maxima                                            # Store the maxima and minima from the training set
        self.minima = lr.minima
        
    def classify(self,df_test):
        """ Classifying the inputs into the respective class through threshold"""
        
        for col in range(len(df_test.columns[1:-1])):                      # Normalising the test values using train max , min
            maximum = self.maxima[col]
            minimum = self.minima[col]
            df_test.iloc[:,col + 1] = (df_test.iloc[:,col+1] - minimum)/(maximum-minimum)
            
        df_test['y_hat'] = maths.matmul(df_test.iloc[:,:-1],self.w_star)   # Predicting
        df_test['y_hat'] = df_test['y_hat'].apply(lambda x : 1 if x > self.threshold else 0) 
                                                                           # Classifying through threshold
        
        self.create_confusion_matrix(list(df_test['y']),list(df_test['y_hat']))


    def create_confusion_matrix(self,y,y_hat):
        """ Creating the confusion matrix """
        
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
        """ Finding Precision """
        
        confusion_matrix = self.confusion_matrix
        precision = round((confusion_matrix['tp'])/(confusion_matrix['fp']+confusion_matrix['tp'])*100,2) 
        if show : print("Precision : ",precision,"%")
        return precision

    def find_accuracy(self,show = True):
        """ Finding Accuracy """
        
        confusion_matrix = self.confusion_matrix
        accuracy = round((confusion_matrix['tp']+confusion_matrix['tn'])/(confusion_matrix['tp'] + confusion_matrix['tn'] + confusion_matrix['fp'] + confusion_matrix['fn'])*100,2)
        if show : print("Accuracy : ",accuracy,"%")
        return accuracy

    def find_sensitivity(self,show = True):
        """ Finding Sensivity """
        
        confusion_matrix = self.confusion_matrix
        sensitivity = round((confusion_matrix['tp'])/(confusion_matrix['tp']+confusion_matrix['fn'])*100,2)
        if show : print("Sensivity : ",sensitivity,"%")
        return sensitivity

    def find_specificity(self,show = True):
        """ Finding Specificity """
        
        confusion_matrix = self.confusion_matrix
        specificity = round((confusion_matrix['tn'])/(confusion_matrix['fp']+confusion_matrix['tn'])*100,2)
        if show : print("Specificity : ",specificity,"%")
        return specificity

    def find_fscore(self,show=True):
        """ Finding FScore """
        
        confusion_matrix = self.confusion_matrix
        f_score = round(2/((1/self.find_precision(False))+(1/self.find_sensitivity(False))),2)
        if show : print("F1 Score : ",f_score)
        return f_score        

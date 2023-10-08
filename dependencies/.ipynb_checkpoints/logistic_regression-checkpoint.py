import pandas as analytics
import numpy as maths
import matplotlib.pyplot as graph
import os
from math import exp, log
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
        df_data.columns = names        # read data with names
        df_data['x0'] = 1                                                    # append 1 column
        df_data = df_data[['x0'] + names]
        df_data['y'] = df_data['y'].replace(-1,0)                            # preprocessing
        self.df_data = df_data
        
    def split_dataset(self,validation_perc, training_perc ):
        """ Split the dataset into validation set and test set according to the validation_perc. Training part from the validation set is determined by training_perc. So training percentage is training_perc of validation_perc and testing set is 1 - validation_perc """
        
        validation_number = int(len(self.df_data)*validation_perc)          # Validation Number
        self.training_perc = training_perc
        
        self.df_data = self.df_data.sample(frac = 1)                        # Shuffling data
        df_validation = self.df_data[:validation_number].reset_index(drop=True)                    # Distributing data into validation and test
        df_test = self.df_data[validation_number:].reset_index(drop = True)                          
        
        return df_validation, df_test
    
    def find_weights(self,df_validation, alphas , number_of_iterations, process = 'logistic' ):
        """ Using linear or logistic regression to find the weights for setting the hyperplane """
        self.df_validation = df_validation
        if process == 'linear' : self.linear_regression_weights(alphas,number_of_iterations)
        elif process == 'logistic' : self.logistic_regression_weights(alphas, number_of_iterations)
        del self.training_perc
        del self.df_validation
            

    def linear_regression_weights(self,alphas, number_of_iterations):
        
        self.df_validation.iloc[:,1:].to_csv('validation.csv',header=False,index=False)
                                                                            # Saving the validation set to take as input in lin reg
        lr = linear_regression()
        lr.load_data('validation.csv')
        os.remove('validation.csv')
        lr.monte_carlo(alphas,number_of_iterations,self.training_perc)
        df_validation_training , df_validation_testing = lr.split_data()
        lr.train(df_validation_training, sgd = True, plot_rmse = False, plot_metrics = True)
        lr.test(df_validation_testing)
        
        
        self.w_star = lr.w_star                                            # Store the optimal weights from lin reg
        self.maxima = lr.maxima                                            # Store the maxima and minima from the training set
        self.minima = lr.minima


    def logistic_regression_weights(self,alphas, number_of_iterations):
        self.alpha = self.monte_carlo( alphas,number_of_iterations)
        
        df_validation = self.df_validation
        df_validation = df_validation.sample(frac=1)
        training_number = int(self.training_perc * len(df_validation))

        df_validation_training = df_validation[:training_number]
        df_validation_testing = df_validation[training_number:]

        df_validation_training , self.maxima, self.minima = self.normalise(df_validation_training)
        
        X = df_validation_training.iloc[:,:-1].reset_index(drop=True)
        Y = df_validation_training.iloc[:,-1].reset_index(drop=True)

        self.w_star , norm_length = self.stochastic_gradient_descent(self.alpha,X,Y)
        
        
        

    def stochastic_gradient_descent(self,alpha,X,Y):
        
        epsilon = 5e-3
        w_star = maths.matrix(maths.random.rand(X.shape[1])).T
        w_old = w_star + [1]
        
        liklihoods = []
        norms = [1000,100]
        
        while norms[-1] != 0 and (norms[-2]/norms[-1] - 1) > epsilon :
            w_old = w_star
            loss = []
            for i in range(len(X)):
                x = maths.matrix(X.iloc[i]).T
                y = Y.iloc[i]
                y_hat = self.sigmoid(x,w_star)
                w_star = w_star + alpha * (y - y_hat) * x
                loss.append(self.liklihood(x,y,w_star))
            norms.append(self.norm( w_star - w_old ))
            if len(norms) > 2: differential = norms[-2] - norms[-1]
            liklihoods.append(sum(loss))
        return w_star, len(norms)
        

    def monte_carlo(self, alphas, number_of_iterations):
        df_validation = self.df_validation
        epsilon = 1e-3
        training_number = int(len(df_validation) * self.training_perc)
        number_of_iterations = 3
        accuracies_list = []
        iterations_list = []
        
        for iteration_number in range(1,number_of_iterations+1):
            print("Iteration Number :",iteration_number)
            print("---------------------")
        
            alpha_accuracy = {}
            alpha_iterations = {}
            
            for alpha in alphas:
                print("  For alpha : ",alpha, end = " ")
                    
                df_validation = df_validation.sample(frac=1)
                df_validation_train = df_validation.iloc[:training_number]
                df_validation_test = df_validation.iloc[training_number:]
            
                df_validation_train = df_validation_train.sample(frac=1)
                X = df_validation_train.iloc[:,:-1].reset_index(drop=True)
                Y = df_validation_train.iloc[:,-1].reset_index(drop=True)
            
                w_star, norm_length = self.stochastic_gradient_descent(alpha, X, Y)   
                
                df_validation_test['y_hat'] = df_validation_test.iloc[:,:-1] @ w_star   # Predicting
                df_validation_test['y_hat'] = df_validation_test['y_hat'].apply(lambda x : 1 if x > self.threshold else 0) 
            
                y = df_validation_test['y'].reset_index(drop=True)
                y_hat = df_validation_test['y_hat'].reset_index(drop = True)
            
                tp = 0
                tn = 0
                for i in range(len(y)):
                    if y[i] == y_hat[i] == 1:
                        tp = tp + 1
                    elif y[i] == y_hat[i] == 0:
                        tn = tn + 1
                    
                accuracy = round((tp+tn)/(len(df_validation_test))*100,2)
                print("    Accuracy : ",accuracy, end = " ")
                print("    Number of iterations :",norm_length)
        
                alpha_accuracy.update({alpha:accuracy})
                alpha_iterations.update({alpha:norm_length})
                
            accuracies_list.append(alpha_accuracy)
            iterations_list.append(alpha_iterations)
            print()
        
        df_accuracies = analytics.DataFrame(accuracies_list)
        df_iterations = analytics.DataFrame(iterations_list)
        
        if df_accuracies.mean().all() :
            alpha = df_iterations.mean().idxmin()
        else : alpha = df_accuracies.mean().idxmax()
        
        return alpha



    def normalise(self,df_train):
        maxima = []
        minima = []
        for col in df_train.columns[1:-1]:
            maximum = df_train[col].max()
            minimum = df_train[col].min()
            diff = maximum - minimum

            df_train[col] = (df_train[col] - minimum) / diff
            maxima.append(maximum)
            minima.append(minima)
        return df_train, maxima, minima
        
    
    
    def classify(self,df_test):
        """ Classifying the inputs into the respective class through threshold"""
        
        for col in range(len(df_test.columns[1:-1])):                      # Normalising the test values using train max , min
            maximum = self.maxima[col]
            minimum = self.minima[col]
            df_test.iloc[:,col + 1] = (df_test.iloc[:,col+1] - minimum)/(maximum-minimum)
            
        df_test['y_hat'] = df_test.iloc[:,:-1] @ self.w_star   # Predicting
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

        matrix = [[tp,fp],[fn,tn]]
        figure = graph.figure(figsize=(15,10))
        graph.matshow(maths.matrix(matrix))
        graph.colorbar()
        graph.show()


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

    def norm(self,vector):
        """2 norm"""
        vector = list(vector)
        return float(sum([z**2 for z in vector]))**0.5
    
    def sigmoid(self,x,w):
        """ Sigmoid Function """
        if -w.T@x < 1e-20 : w_x = exp(-w.T@x)
        else : w_x = 1
        return 1/(1 + w_x)

    def liklihood(self,x,y,w):
        """ Liklihood Function """
        y_hat = self.sigmoid(x,w)
        if y_hat == 1: return y*log(y_hat)
        else : return y*log(abs(y_hat)) + (1-y)*log(abs(1-y_hat))

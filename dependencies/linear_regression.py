from __future__ import annotations
from typing import Annotated
import pandas as analytics
import numpy as maths
import matplotlib.pyplot as graph
import warnings
warnings.simplefilter("ignore")




class linear_regression :
    def __init__(self):
        self.N = 0                   # Number of datapoints in the dataset
        self.n = 0                   # Number of attributes in the dataset
        self.maxima = []             # To store the maxima of each attribute of the training set
        self.minima = []             # To store the minima of each attribute of the testing set
        self.sgd = True              # To run Stochastic Gradient Descent or not
        self.plot_rmse = False       # To plot the RMSE graph or not
        self.plot_metrics = False    # To plot the metrics asked in the Assignment questions or not
        self.alpha = 0               # The optimal hyperparameter
                                     #
    def help(self):
        "Help Function to give the details of the Class"
        
        print("""
        
    Variables
    =========
        N              # Number of datapoints in the dataset
        n              # Number of attributes in the dataset
        maxima         # To store the maxima of each attribute of the training set
        minima         # To store the minima of each attribute of the testing set
        sgd            # To run Stochastic Gradient Descent or not
        plot_rmse      # To plot the RMSE graph or not
        plot_metrics   # To plot the metrics asked in the Assignment questions or not
        alpha          # The optimal hyperparameter
        df_data        # The preprocessed complete dataset
        w_star         # Optimal Weights selected


    Functions
    =========
        load_data( path )     
        # Reads the data from the path and Preprocesses it

        preprocess_data()   
        # Pre-Process the data by Appending x0 column of 1s , Replacing Null values with 0 etc.

        normalise_data( df )  
        # Normalise the data using Min-Max method

        rmse( y , y_hat )       
        # RMSE between Actual Output and Predicted Output

        cost( y , y_hat )       
        # Find the cost associated with the current parameter

        optimal_weight( df )  
        # Use the function to find the optimal weights using different values of alphas and a option with the user to select from either batch gradient method or stochastic gradient method. Aditionally, all the metrics required such as cost vs iteration, cost vs weights, hyperplane etc. will be plotted using this function according to the users requirement. This function basically checks whether the w found either through batch or stochastic gradient converges or not i.e. whether it is less than epsilon

        batch_gradient_descent( w_star , df , alpha ) 
        # Training the weights after each epoch is batch gradient descent

        stochastic_gradient_descent( w_star , df , alpha ) 
        # Training the weights within each epoch is stochastic gradient descent

        monte_carlo( alphas , k , training_perc ) 
        # Running the optimal weight function k times for cross validating the data fit in Linear Gradient also through this method, optimal alpha from a series of alpha is found such that the alpha has the least avg rmse in all the iterations
        
        plot_required_metrics(self,errors, costs, weights,w_star)   
        # According to the question all the metrics that were needed to be plotted are plotted in this function

        train( df_train , sgd = True , plot_rmse = True , plot_metrics = True )
        # Train the data for the chosen alpha, with an option to whether to use sgd , plot the rmse, question metrics etc."

        test( df_test )
        # Predict the output through the learnt w_star and check its performance error

        split_data():
        # Splits the whole dataset into Training and Testing set. Returns df_train and df_test


    Sequence of Function Calls 
    ===========================
        # load_data( path )
        # monte_carlo( alphas , k , training_perc )
        # training_set , testing_set = split_data()
        # train( training_set , sgd , plot_rmse , plot_metrics )
        # test

        """)
                            


    def load_data(self,path):   
        "Reads the data from the path and Preprocesses it"

        df_data = analytics.read_csv(path)
        n = df_data.shape[1] - 1
        names = ["x"+str(i+1) for i in range(n)] + ['y']
        self.df_data = analytics.read_csv(path, names = names)

        
        self.preprocess_data()
        self.N , n = self.df_data.shape
        self.n = n-1


    def preprocess_data(self):
        "Pre-Process the data by Appending x0 column of 1s , Replacing Null values with 0 etc." 

        df = self.df_data
        df['x0'] = 1
        cols = list(df.columns)
        cols.insert(0,cols.pop())
        df = df[cols]

        df = df.fillna(0)
        self.df_data = df            # To store the whole preprocessed data 
        
                                    

    def normalise_data(self,df):
        "Normalise the data using Min-Max method"

        maxima = self.maxima
        minima = self.minima
        cols = list(df.columns[1:-1])
        
        if maxima and minima :
#             when testing
            
            for col in cols:
                ind = cols.index(col)
                maximum_value = maxima[ind]
                minimum_value = minima[ind]
                diff = maximum_value - minimum_value
                df[col] = (df[col] - minimum_value) / diff
                    
        else :
#             when training
            for col in cols:
                maximum_value = max(df[col])
                minimum_value = min(df[col])
                diff = maximum_value - minimum_value
                df[col] = (df[col] - minimum_value) / diff

                maxima.append(maximum_value)
                minima.append(minimum_value)
                self.maxima = maxima
                self.minima = minima

        return df
        

    def rmse(self,y,y_hat):
        "Find the Root Mean Square Error"

        y_error = y - y_hat
        y_error = y_error ** 2
        length = len(y_error)
        y_error = sum(y_error)/length
        y_error = y_error ** 0.5
        return y_error



    def cost(self,y,y_hat):
        "Find the cost associated with the current parameter"

        error = (y - y_hat) ** 2
        error = error / 2
        return sum(error)


    def optimal_weight(self,df):
        """Use the function to find the optimal weights using different values of alphas and a option with the user to
        select from either batch gradient method or stochastic gradient method. 
        Additionally, all the metrics required such as cost vs iteration, cost vs weights, hyperplane etc. will
        be plotted using this function according to the users requirement. 
        This function basically checks whether the w found either through batch or stochastic gradient converges or not
        i.e. whether it is less than epsilon"""

        if self.alpha : alphas = [self.alpha]
        else :alphas = self.alphas
        
        n = self.n
        
        rmses = {}

        for alpha in alphas :
            w_star = []
            for i in range(n): w_star.append([maths.random.randint(1,100)])
            w_star = maths.matrix(w_star)
            costs = []
            errors = []
            weights = []
            norm = 1

            epsilon = 1e-6

            while norm > epsilon :

                if self.sgd : w_star , norm , y_hats = self.stochastic_gradient_descent(w_star, df, alpha)
                else : w_star , norm , y_hats = self.batch_gradient_descent(w_star, df, alpha)

                errors.append(self.rmse(df['y'],y_hats))
                costs.append(self.cost(df['y'],y_hats))
                weights.append(w_star)

            if self.plot_metrics :
                self.plot_required_metrics(errors, costs, weights,w_star)

            if self.plot_rmse :    
                figure = graph.figure(figsize=(20,10))
                graph.title("RMSE vs Iter_Number for alpha = "+str(alpha))
                graph.plot([loop+1 for loop in range(len(errors))],errors)

            rmses.update({alpha:errors})

        return w_star,rmses



    def batch_gradient_descent(self,w_star,df,alpha):
        "Training the weights after each epoch is batch gradient descent"

        n = self.n
    
        differentials = []
        y_hats = []

        for row in range(len(df)):
            x = maths.matrix(df.iloc[row][:n]).T
            y_hat = float(maths.matmul(w_star.T,x))

            differential = (float(df.iloc[row]['y']) - y_hat) * x
            differentials.append(differential)
            y_hats.append(y_hat)

        df['differentials'] = differentials
        grad_sum = alpha * df['differentials'].sum()

        w_star = w_star + grad_sum

        norm = maths.square(grad_sum)
        norm = abs(norm.sum())

        return w_star , norm , y_hats


    def stochastic_gradient_descent(self,w_star,df,alpha):
        "Training the weights in each epoch is stochastic gradient descent"

        n = self.n
    
        differentials = []
        y_hats = []

        for row in range(len(df)):
            x = maths.matrix(df.iloc[row][:n]).T
            y_hat = float(maths.matmul(w_star.T,x))
            differential = (float(df.iloc[row]['y']) - y_hat) * x

            w_star = w_star + alpha * differential

            differentials.append(differential)
            y_hats.append(y_hat)

        df['differentials'] = differentials
        grad_sum = alpha * df['differentials'].sum()

        norm = maths.square(grad_sum)
        norm = abs(norm.sum())

        return w_star , norm , y_hats



    def monte_carlo(self,alphas,k,training_perc):
        """Running the optimal weight function k times for cross validating the data fit in Linear Gradient
        also through this method, optimal alpha from a series of alpha is found such that the alpha has the least avg
        rmse in all the iterations"""

        self.alphas = alphas
        iteration_counts = [] 
        self.training_size = int(training_perc * len(self.df_data))
        df_data = self.df_data.copy()
        
        training_size = self.training_size
        
        for i in range(k) :
            df_value = {}
            df_data = df_data.sample(frac = 1)
            df_train = df_data[0:training_size]

            df_train = self.normalise_data(df_train)
            self.minima = []
            self.maxima = []
            print("For iteration number ",i+1," : ")
            w, rmses = self.optimal_weight(df_train)

            for alpha in alphas :
                rmse_value = maths.array(rmses[alpha]).mean()
                df_value.update({alpha:rmse_value})
                print("\tAverage RMSE for alpha : ",alpha," : " , rmse_value)
            iteration_counts.append(df_value)
            print()
        df_monte_carlo = analytics.DataFrame(iteration_counts)
        df_monte_carlo.plot(figsize=(40,20),title = "Number of Iterations vs Alpha")
        alpha = (df_monte_carlo.iloc[:,:].mean()).idxmin()
        print("Best Alpha is ", alpha)
        self.alpha = alpha
        
        return alpha


    def plot_required_metrics(self,errors, costs, weights,w_star):
        "According to the question all the metrics that were needed to be plotted are plotted in this function"

        figure = graph.figure(figsize=(20,10))
        graph.title("Cost vs Iterations")                             # Cost vs Iterations for both data
        graph.plot([loop+1 for loop in range(len(errors))],costs)

        if len(w_star) == 2 :
            x1 = [weights[row][0] for row in range(len(weights))]
            x2 = [weights[row][1] for row in range(len(weights))]

            figure = graph.figure(figsize = (25,15))
            axes = graph.axes(projection='3d')
            scattered = axes.scatter3D(x1, x2, costs)
            graph.title("Cost vs Weights")                             # Cost vs Weights for Data 1


            figure = graph.figure(figsize = (25,15))
            graph.title("Hyperplane")                                  # Hyperplane for both Data 1
            x = maths.random.rand(100)
            y = [float(w_star[0]) + float(w_star[1])*x[i] for i in range(len(x))]
            graph.plot(x,y)

        else :

            figure = graph.figure(figsize = (25,15))
            graph.title("Hyperplane")                                  # Hyperplane for both Data 2
            x1 = maths.random.rand(200)
            x2 = maths.random.rand(200)
            y = [float(w_star[0]) + float(w_star[1])*x1[i] + float(w_star[2])*x2[i] for i in range(len(x1))]
            axes = graph.axes(projection='3d')
            scattered = axes.scatter3D(x1, x2, y)


    def train(self, df_train, sgd = True, plot_rmse = True, plot_metrics = True ):
        "Train the data for the chosen alpha, with an option to whether to use sgd , plot the rmse, question metrics etc."

        self.sgd = sgd
        self.plot_rmse = plot_rmse
        self.plot_metrics = plot_metrics
        
        df_train = self.normalise_data(df_train)
        
        w_star , self.train_rmses = self.optimal_weight(df_train)
        avg_rmse = maths.array(self.train_rmses[self.alpha]).mean()
        print("Trained Parameters are : ",w_star)
        print("Average training RSME for alpha = ",self.alpha," : ",avg_rmse)
        
        self.w_star = w_star            # Optimal Weight

        
    def test(self,df_test):
        "Predict the output through the learnt w_star and check its performance error"

        df_test = self.normalise_data(df_test)
        df_test['y_hat'] = maths.matmul(df_test.iloc[:,:self.n],self.w_star)
        
        self.test_rmses = maths.array(self.rmse(df_test['y'],df_test['y_hat']))
        print("Average testing RMSE for alpha = ",self.alpha," : ",self.test_rmses)

        df_final = analytics.DataFrame()
        df_final['y'] = df_test['y'].sort_values().reset_index(drop=True)
        df_final['y_hat'] = df_test['y_hat'].sort_values().reset_index(drop=True)

        df_final.plot(y=['y'],figsize=(20,10),c='b',title="Y")
        df_final.plot(y=['y_hat'],figsize=(20,10),c='r',title="Y Predicted")

        
    def split_data(self):
        "Splits the whole dataset into Training and Testing set. Returns df_train and df_test"
        
        df_data = self.df_data.sample(frac=1)
        df_train = df_data[0:self.training_size]
        df_test = df_data[self.training_size : ]    

        return df_train , df_test
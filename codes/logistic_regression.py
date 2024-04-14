import pandas as analytics
import numpy as maths
from math import log

class logistic_regression:
    def __init__(self,data,hyperparameters):
        self.hyperparameters = hyperparameters
        df_raw = data
        train_data = int(0.8*len(df_raw))
        df_raw=df_raw.sample(frac=1)
        self.datapoints=df_raw.shape[0]
        self.attributes=df_raw.shape[1]-1
        self.classes=df_raw[df_raw.columns[-1]].unique()
        self.df_train=df_raw[:train_data]
        self.df_test=df_raw[train_data:]
        try:
            print("\n\n")
            if len(df_raw.select_dtypes(object).columns)>0:
                
                print("Data is not preprocessed...Please pre process it and then supply")
            else:
                print("Data seems as preprocessed and normalised !! Ready to train...")
            
        except TypeError:
            print("Data is not preprocessed ... Please Preprocess it and then supply")
        
    
    def sigma(self,x):
        return 1/(1+maths.exp(-x))
    
    def has_converged(self,losses,loss,epoch):
        converged=(maths.linalg.norm(maths.array(losses)+loss)<self.hyperparameters['tolerance']) | (-loss<self.hyperparameters["tolerance"]) | (epoch>=self.hyperparameters["max_epoch"])
        return converged
    
    def train(self):
        df_train=self.df_train[self.df_train.columns[:self.attributes+1]]
        weights=maths.random.choice(maths.linspace(self.hyperparameters['weights_lb'],self.hyperparameters['weights_ub']),(self.attributes,1))
        print("Initial Weights:",weights.T,"\n")
        alpha=self.hyperparameters['alpha']
        
        loss=-100
        losses=[0]
        epoch=1
        converged=False
        
        while not converged:
            for i in range(len(df_train)):
                data=df_train.iloc[i]
                x=data[:self.attributes]
                y=data['y']
                f_x=self.sigma(weights.T@x)
                loss_gradient=maths.matrix((y-f_x)*x).reshape((self.attributes,1))
                weights=weights+self.hyperparameters['alpha']*loss_gradient
            
            loss=y*log(f_x)+(1-y)*log(1-f_x)
            print("Epoch #",str(epoch).zfill(3),":: Loss :%.3f"%-loss)
                
            epoch=epoch+1
            losses.append(-loss)
            if epoch>5:
                losses=losses[-5:]
            
            converged=self.has_converged(losses,loss,epoch)
        
        print("\nFinal Weights:",weights.T)
        self.weights=weights
        
    

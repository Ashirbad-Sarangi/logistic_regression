import pandas as analytics
import numpy as maths


def normalize(df_raw):
    for col in df_raw.columns:
        if analytics.api.types.is_numeric_dtype(df_raw[col].dtype):
            maximum=df_raw[col].max()
            minimum=df_raw[col].min()
            df_raw[col]=(df_raw[col]-minimum)/(maximum-minimum)
    return df_raw

def get_summary(df_raw):
    print("Data Summary")
    print("============")
    datapoints=df_raw.shape[0]
    attributes=df_raw.shape[1]-1
    classes=df_raw[df_raw.columns[-1]].unique()
    print("No.of Datapoints:",datapoints)
    print("No.of Attributes:",attributes)
    print("Unique y Values(RAW):",classes)
    if len(classes)==2: print("Binary Class Data")
    else: print("Multi Class Data")
    
    print("\nData Statistics")
    print("===============")
    print(df_raw.describe().T)
    
    return df_raw

def check_performance(df, weights, pos_class, neg_class ):
    classification = df[df.columns[:weights.shape[0]]] @ weights
    df['classification'] = classification
    df['classification'] = df['classification'].apply(lambda x: 1 if x > 0.5 else 0 )
    actual = df['y']
    computed = df['classification']
    
    
    print("METRICS")
    print("=======")
    print("Accuracy : %.3f"% accuracy(actual,computed,pos_class,neg_class))
    print("Sensitivity :%.3f"% sensitivity(actual,computed,pos_class,neg_class))
    print("Specificity :%.3f"% specificity(actual,computed,pos_class,neg_class))
    print("Precision :%.3f"% precision(actual,computed,pos_class,neg_class))
    print("FMeasure :%.3f"%fmeasure(actual,computed,pos_class,neg_class))



def plot_graph(df_raw):
    pos_df=df_raw[df_raw['y']==1]
    neg_df=df_raw[df_raw['y']==0]
    graph.scatter(pos_df['x1'],pos_df['x2'],color='blue',label='positive')
    graph.scatter(neg_df['x1'],neg_df['x2'],color='red',label='negative')
    graph.xlabel('x1')
    graph.ylabel('y1')
    graph.legend()
    

def accuracy(actual,computed,pos_class,neg_class):
    """Computed Positive"""
    try:
        tp=computed[computed==actual].value_counts()[pos_class]
    except KeyError:
        print("No true positives...",end="")
        tp=0
    try:
        tn=computed[computed==actual].value_counts()[neg_class]
    except KeyError:
        print("No true negatives...",end="")
        tn=0
    total=len(actual)
    accuracy=(tp+tn)/total
    return accuracy

def sensitivity(actual,computed,pos_class,neg_class):
    """Ratio of Belongs to true class and classified true and belongs to true"""
    try:
        tp=computed[computed==actual].value_counts()[pos_class]
    except KeyError:
        print("No true positives...",end="")
        tp=0
    total=actual.value_counts()[pos_class]
    sens=(tp)/total
    return sens

def specificity(actual,computed,pos_class,neg_class):
    """Ratio of Belongs to false class and classified false and belongs to false"""
    try:
        tn=computed[computed==actual].value_counts()[neg_class]
    except KeyError:
        print("No true negatives...",end="")
        tn=0
    total=actual.value_counts()[neg_class]
    specs=(tn)/total
    return specs

def precision(actual,computed,pos_class,neg_class):
    """Ratio of Belongs to true and classified true class"""
    try:
        tp=computed[computed==actual].value_counts()[pos_class]
    except KeyError:
        print("No true positives...",end="")
        tp=0
        
    try:
        tn=computed[computed==actual].value_counts()[neg_class]
    except KeyError:
        print("No true negatives...",end="")
        tn=0
    prec=tp/(tp+tn)
    return prec

def fmeasure(actual,computed,pos_class,neg_class):
    """Harmonic Mean of Precision and Sensitivity"""
    pr=precision(actual,computed,pos_class,neg_class)
    sens=sensitivity(actual,computed,pos_class,neg_class)
    
    fmeas=2*pr*sens/(pr+sens)
    return fmeas
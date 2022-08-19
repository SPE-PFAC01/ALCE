#!/usr/bin/env python
# coding: utf-8

# # Virtual Flow Meter using critical flow data at production choke
# ### Data and methodologies mentioned in reference below are reproduced here.
# ##### Reference: Barjouei, H.S., Ghorbani, H., Mohamadian, N. et al. Prediction performance advantages of deep machine learning algorithms for two-phase flow rates through wellhead chokes. J Petrol Explor Prod Technol 11, 1233–1261 (2021). https://link.springer.com/article/10.1007/s13202-021-01087-4 [Accessed 3 Jun 2021]
# 

# ### Load Python Libraries

# In[1]:


# Data storage, exploration
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Data imputing
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# ML Model libraries

# The following three lines allow multiple and non-truncated outputs 
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
pd.set_option('display.max_columns', None)


# # Import Data & Preliminary Data Exploration
# 1. How many data records? 
# 2. How many variables / features or columns in each data record?
# 3. Peek at the first five records and the last five records

# In[11]:


# Import Data
vfm = pd.read_csv(r"C:\Users\RajanChokshi\OneDrive - Accutant Solutions LLC\Accutant RC\Business\Training\2022\ALCE Training\ChokeFlow Example\SorushDatasetChokeFlow.csv")
vfm.columns = ['SampleID', 'Well', 'D64', 'Pwh', 'γo', 'GLR', 'QL']
                  #names=columns, )


# In[12]:


vfm.info()
vfm.head()
vfm.tail()


# ## Statistics on each column / feature / variable
# 1. What are NaNs in the first five records for <b>chokegaslift</b>? There could be more in other records for this variable... and for other(s).
# 2. Review how many measurements for each variable in the <b>count</b> row. Why chokegaslift has less measurements? NaNs? How many? 
# 3. Review Mean, min, std dev and percentile values for each variable. 
#     1. Could min-values be negative for <b>chokeprod, dhp</b>? 
#     2. How about min-values being zero for certain variables?
#     3. What are the median values?
# 4. What does it mean when 25% percentile-value is 0.0 for <b>dhp, gasliftpressure, dht</b> variables?

# In[29]:


# plot statistics
plt.figure(figsize=(15,6));
plt.yscale("log");
plt.grid('y');
_ = plt.xticks(rotation='vertical');
sns.boxplot(data=vfm.drop(columns=['Sample #', 'Well']))


# #### Boxplot Visualization
# In the visualization above, why some boxes are very tall (long color bars).
# 1. Which variable has the smallest distribution?
# 2. Which variable is widely distributed?
# 3. What does it mean when one whisker is longer than the other?

# In[30]:


plt.figure(figsize=(15,6));
plt.grid('y');
sns.boxplot(data=vfm, x="Well", y="QL, STB/D");
plt.figure(figsize=(15,6));
sns.boxplot(data=vfm, x="Well", y="GLR, scf/STB");


# ### Histograms
# Help visualize how measurements are distributed.
# Wouldn't we like them to be normally distributed?!?

# In[28]:


vfm.drop(columns=['SampleID']).hist(figsize=(20, 20));

# In[31]:


vfm.columns


# In[32]:


plot_vars = ['D (1/64in)', 'Pwh, Psig', 'γo', 'GLR, scf/STB', 'QL, STB/D']
# Define a function to plot histogram and scatterplot for the specified variables/columns of provided dataframe 
def plotPairgrid(df, plot_vars):
    g = sns.PairGrid(data=df, vars=plot_vars, diag_sharey=False);
    g.map_upper(sns.scatterplot, s=15);
    g.map_lower(sns.kdeplot);
    g.map_diag(sns.kdeplot, lw=2);

plotPairgrid(vfm, plot_vars);


# ### Correlations between Variables
# Let's find out if two variables are correlated by calculating correlation coefficients between two variables. 
# 1. Positive value (positive correlation) means one increases with another in the dataset; and 
# 2. Negative value (negative correlation) means one decreases while another increases and vice versa. 
# 3. Magnitude of the correlation coefficient indicates strength of the correlation.
# 
# #### Why do we want to perform this exercise?

# In[34]:

# Pearson Correlation Coefficient
# Pearson correlation assumes that the data we are comparing is normally distributed. 
# When that assumption is not true, the correlation value is reflecting the true association.
pearson = vfm[plot_vars].corr(method='pearson')
g1 = sns.heatmap(pearson, center=0.0, linewidths=0.1, square=True, annot=True, vmin=-1, vmax=1., fmt='1.2f')
g1.set_xticklabels(g1.get_xticklabels(), rotation=30);
g1.set_title('Heatmap')

# Spearman Rank Correlation
# Spearman correlation does not assume that data is from a specific distribution, 
# so it is a non-parametric correlation measure. Spearman correlation is also known as Spearman’s rank correlation as it computes correlation coefficient on rank values of the data.
spearman = vfm[plot_vars].corr(method='spearman')

#Spearman’s correlation coefficients shown above reveal that the input variables,
#    GLR and γo are inversely related to QL, 
#    Pwh and D64 display positive correlations with QL. 
#    D64 shows the lowest correlation coefficient with QL of the four input variables evaluated.
#    Prevailing flow through the wellhead chokes of the Sorush oil field conforms to a critical flow regime

# # Data Exploration
# ### Missing Data at Macro Level
# Are there any null (NaN) measurements for numeric data columns?

# In[36]:


# Q1. How many nulls are there?
vfm.isnull().sum()

# Q2. How many values are zero or -ve
(vfm[plot_vars] <= 0.0).sum()


# ### Prepare features and target data arrays 
# Also separate in training and test sets

# In[68]:


# Separate targets from inputs
target= ['QL, STB/D']

y = vfm[target].values.ravel()
X = vfm[plot_vars].drop(columns=target).to_numpy()


# In[70]:


# Do we have nulls in targets and inputs?
np.isnan(y).any(), np.isnan(X).any()
y.shape, X.shape


# In[91]:


# Normalize all the samples between [-1, +1]
X = (X- X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))*2 - 1


# In[93]:


X[:5]


# In[94]:


## Sub-divide datatest into training and testing: 80 - 20% split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, random_state=1002)
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

def calc_predMetrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return [rmse, mape, r2]
    
from sklearn.model_selection import cross_val_score


# In[97]:


#  In this study, the RBF kernel is used with the SVR algorithm 
# to predict two-phase flow rate (Ql) through a wellhead choke.
from sklearn.svm import SVR
svr_rbf = SVR(kernel="rbf", C=100000, gamma=0.05, epsilon=0.1)
y_pred_svr = svr_rbf.fit(X_train, y_train).predict(X_test)
svr_rbf_metric = {"svr_rbf":calc_predMetrics(y_test, y_pred_svr)}
svr_rbf_metric


# In[ ]:





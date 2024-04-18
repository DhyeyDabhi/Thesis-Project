import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import missingno as msno
import pickle
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score

def make_facies_log_plot(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(12, 6))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.40')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
    ax[4].plot(logs.PE, logs.Depth, '-', color='black')
    im=ax[5].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((5*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("ILD_log10")
    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI")
    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND")
    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[4].set_xlabel("PE")
    ax[4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[5].set_xlabel('Facies')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)
    st.pyplot(f)

def replace_outliers(df, method='median'):
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if method == 'Median Imputation':
            replacement_value = df[column].median()
        elif method == 'Mean Imputation':
            replacement_value = df[column].mean()
        else:
            raise ValueError("Method must be 'median' or 'mean'")
        
        df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound),
                              replacement_value, df[column])
        
    return df

st.title("Geological Facies Classification")

st.write("The app takes the Well LOg data and apply the necessary data preprocessing techniques, clean the data, calculate the Geological Facies and calculate the performance metrics for the specific model applied.")
st.write("The data should be in the CSV (.csv) format and should contain the following columns: Facies, Formation, Well Name, Depth, GR, ILD_log10, DeltaPHI, PHIND, PE, NM_M, RELPOS")

uploaded_file = st.file_uploader("Choose the well log file: ")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Missing Data Matrix")
        fig, ax = plt.subplots()
        msno.matrix(df, ax=ax, sparkline=False, color=(0.2, 0.3, 0.5))
        # Set specific font sizes
        ax.set_title('Missing Data Matrix', fontsize=12)  # Set title font size
        ax.tick_params(axis='x', labelsize=8)  # Set font size of the x-axis tick labels
        ax.tick_params(axis='y', labelsize=8) 
        st.pyplot(fig)
    except:
        st.write("An Error has occured!")

try:
    x = df[['GR','ILD_log10','DeltaPHI','PHIND','PE','NM_M','RELPOS','Depth']]
except:
    st.write('Please provide all the required data fields!')
    st.stop()

# Handle missing values
int_tech = st.selectbox("Handle Missing values using: ",['Median Imputation','Mean Imputation'],key='int_tech')
if int_tech=='Median Imputation':
    x_filled = x.fillna(x.median())

if int_tech=='Mean Imputation':
    x_filled = x.fillna(x.mean())

st.write('After filling the missing values:')
fig, ax = plt.subplots()
msno.matrix(x_filled, ax=ax, sparkline=False, color=(0.2, 0.3, 0.5))
# Set specific font sizes
ax.set_title('Missing Data Matrix', fontsize=12)  # Set title font size
ax.tick_params(axis='x', labelsize=8)  # Set font size of the x-axis tick labels
ax.tick_params(axis='y', labelsize=8) 
st.pyplot(fig)

#Handle outliers
impt_tech = st.selectbox("Handle Missing values using: ",['Median Imputation','Mean Imputation'])
x_out_hand = replace_outliers(x_filled,method=impt_tech)

with open('./Facies-Classification-Machine-Learning/scaler.pkl','rb') as file:
    scaler = pickle.load(file)
x_scaled = x_filled
x_scaled[['GR','ILD_log10','DeltaPHI','PHIND','PE','NM_M','RELPOS']] = scaler.transform(x_filled[['GR','ILD_log10','DeltaPHI','PHIND','PE','NM_M','RELPOS']])

model_name = st.selectbox('Select a ML model: ',['Decision Tree Classifier','Gaussian Process Classifier','K Nearest Neighbors','Logistic Regression','Neural Network Model','Random Forest Classifier','Support Vector Machine'])
if model_name=='Decision Tree Classifier':
    with open('./Facies-Classification-Machine-Learning/DTC.pkl','rb') as file:
        model = pickle.load(file)
    yhat = model.predict(x_scaled.drop(columns='Depth'))

if model_name=='Gaussian Process Classifier':
    with open('./Facies-Classification-Machine-Learning/GPC.pkl','rb') as file:
        model = pickle.load(file)
    yhat = model.predict(x_scaled.drop(columns='Depth'))

if model_name=='K Nearest Neighbors':
    with open('./Facies-Classification-Machine-Learning/KNN.pkl','rb') as file:
        model = pickle.load(file)
    yhat = model.predict(x_scaled.drop(columns='Depth'))

if model_name=='Logistic Regression':
    with open('./Facies-Classification-Machine-Learning/LR_model.pkl','rb') as file:
        model = pickle.load(file)
    yhat = model.predict(x_scaled.drop(columns='Depth'))

if model_name=='Neural Network Model':
    with open('./Facies-Classification-Machine-Learning/NNC.pkl','rb') as file:
        model = pickle.load(file)
    yhat = model.predict(x_scaled.drop(columns='Depth'))

if model_name=='Random Forest Classifier':
    with open('./Facies-Classification-Machine-Learning/RFC.pkl','rb') as file:
        model = pickle.load(file)
    yhat = model.predict(x_scaled.drop(columns='Depth'))

if model_name=='Support Vector Machine':
    with open('./Facies-Classification-Machine-Learning/SVM.pkl','rb') as file:
        model = pickle.load(file)
    yhat = model.predict(x_scaled.drop(columns='Depth'))

facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00','#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']
x_solln = x_filled.copy()
x_solln['Facies'] = yhat
x_solln['Well Name'] = df['Well Name']
x_filled['Well Name'] = df['Well Name']
x_filled['Facies'] = df['Facies']

well_name = st.text_input("Predict for the well: ",value='SHRIMPLIN')

st.write('Model Predicted Facies: ')
make_facies_log_plot(x_solln[x_solln['Well Name']==well_name],facies_colors=facies_colors)

st.write('Ground Truth Data: ')
make_facies_log_plot(x_filled[x_filled['Well Name']==well_name],facies_colors=facies_colors)

cm = confusion_matrix(df['Facies'],x_solln['Facies'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
st.pyplot()

st.write('Performance of the model is as follows: ')
st.write('Precision of the model: '+str(precision_score(df['Facies'],x_solln['Facies'],average='weighted')))
st.write('Recall of the model: '+str(recall_score(df['Facies'],x_solln['Facies'],average='weighted')))
st.write('F1 score of the model: '+str(f1_score(df['Facies'],x_solln['Facies'],average='weighted')))
st.write('Accuracy of the model: '+str(accuracy_score(df['Facies'],x_solln['Facies'])))











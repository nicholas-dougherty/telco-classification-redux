import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def describe_data(df):
    print('The first three rows are: ')
    print('----------------------------------------------------------')
    print(df.head(3))
    print('----------------------------------------------------------')
    print("The data frame's shape is: ")
    print('-------------------------')
    print(f' Rows: {df.shape[0]} \n Columns: {df.shape[1]}')
    print('-------------------------')   
    print('The data types and column names are: ')
    print(sorted(df))
    print(df.info())
    print('----------------------------------------------------------')   
    print('The summary statistics are as follows: ')
    print('----------------------------------------------------------')
    print(df.describe())
    print('----------------------------------------------------------')      
    print(f'The number of NA\'s is:')
    print('-------------------------')
    print(df.isna().sum())
    print('-------------------------')
    print ('\nMissing values :  ', df.isnull().sum().values.sum())
    print('----------------------------------------------------------')  
    print('Unique Values for the Columns:')
    print('-------------------------')
    limit = 10
    for col in df.columns:
        if df[col].nunique() < limit:
            print(f'Column: {col} \n')
            print(f'Unique Values: {df[col].unique()} \n')
            print(f'Absolute frequencies: \n {df[col].value_counts()} \n')
            print(f'Relative frequencies: \n {df[col].value_counts(normalize=True)} \n')
        else: 
            print(f'Column: {col} \n')
            print(f'Range of Values: [{df[col].min()} - {df[col].max()}] \n')
        print('-----------------------')
    print('-------Done-zo-------------')
    
def plot_target_dist(df):
    sns.set(style = 'whitegrid')
    sns.set_context('paper', font_scale = 2)
    fig = plt.figure(figsize = (20, 10))
    plt.subplot(121)
    plt.pie(df.churn.value_counts(),labels = ['No Churn', 'Churn'], colors="gr",
            autopct = '%.1f%%', radius = 1, textprops={'fontsize': 20, 'fontweight': 'bold'})
    plt.title('Churn Outcome Pie Chart', fontsize = 30, fontweight = 'bold')
    plt.subplot(122)
    t = sns.countplot(df.churn, palette=['#008000','#FF0000'])
    t.set_xlabel('Churn', fontweight = 'bold', fontsize = 20)
    t.set_ylabel('Count', fontweight = 'bold', fontsize = 20)
    plt.title('Churn Outcome Distributions', fontsize = 30, fontweight = 'bold')
    plt.tight_layout()
    
def plot_internet_services(train):
    
    copy = train.copy()
    fig = plt.figure(figsize = (30, 10))

    plt.subplot(131)
    plt.pie(copy.internet_service_type.value_counts(), labels = ['Fiber Optic', 'DSL', 'No Internet'], autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight': 'bold'}, startangle = 180)
    plt.title('Internet Service Composition', fontweight = 'bold', fontsize = 30)
    
    plt.subplot(132)
    copy = copy.groupby('internet_service_type')['churn'].value_counts().to_frame()
    copy = copy.rename({'churn':'pct_total'}, axis = 1).reset_index()
    copy['pct_total'] = (copy['pct_total']/len(train)) * 100
    d = sns.barplot('internet_service_type', y = 'pct_total', hue = 'churn', palette=['#008000','#FF0000'], data = copy)
    d.set_title('% Churn by Internet Service', fontweight= 'bold', fontsize = 30)
    d.set_xlabel('')
    d.set_ylabel('% of Customers', fontweight = 'bold', fontsize = 20)
    d.set(xticklabels = ['DSL', 'Fiber Optic', 'No Internet Service'])
    
   # plt.subplot(133)
   # e = sns.violinplot('internetservice', 'monthlycharges', 'churn', df, split = True)
   # e.set_title('Violin Plot: Monthly Charges by Internet Service', fontweight = 'bold', fontsize = 30)
   # e.set_xlabel('')
   # e.set(xticklabels = ['DSL', 'Fiber Optic', 'No Internet Service'])
   # e.set_ylabel('Monthly Charges($)', fontweight = 'bold', fontsize = 30)

    fig.tight_layout()
    
def plot_services(df):
    copy = df[df.internet_service_type != 'None']
    
    fig = plt.figure(figsize = (40, 15))
    
    plt.subplot(261)
    plt.pie(copy.online_security.value_counts(), labels = ['Yes', 'No'], colors="gr", autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Customers w/ Online Security', fontweight = 'bold', fontsize = 25)
    
    plt.subplot(262)
    plt.pie(copy.online_backup.value_counts(), labels = ['Yes', 'No'], colors="gr", autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Customers w/ Online Backup', fontweight = 'bold', fontsize = 25)
    
    plt.subplot(263)
    plt.pie(copy.device_protection.value_counts(), labels = ['Yes', 'No'], colors="gr", autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Customers w/ Device Protection', fontweight = 'bold', fontsize = 25)
    
    plt.subplot(264)
    plt.pie(copy.tech_support.value_counts(), labels = ['Yes', 'No'], colors="gr", autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Customers w/ Tech Support', fontweight = 'bold', fontsize = 25)
    

def plot_services_churn(df):
    copy = df[df.internet_service_type != 'None']
    
    fig = plt.figure(figsize = (40, 15))
    
    plt.subplot(261)
    copy1 = copy[copy.online_security == 'Yes']
    plt.pie(copy1.churn.value_counts(), labels = ['No Churn', 'Churn'], colors="gr", autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Online Security - Churn %', fontsize = 25, fontweight = 'bold')
    
    plt.subplot(262)
    copy2 = copy[copy.online_backup == 'Yes']
    plt.pie(copy2.churn.value_counts(), labels = ['No Churn', 'Churn'], colors="gr", autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Online Backup - Churn %', fontsize = 25, fontweight = 'bold')
    
    plt.subplot(263)
    copy3 = copy[copy.device_protection == 'Yes']
    plt.pie(copy3.churn.value_counts(), labels = ['No Churn', 'Churn'], colors="gr", autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Device Protection - Churn %', fontsize = 25, fontweight = 'bold')
    
    plt.subplot(264)
    copy4 = copy[copy.tech_support == 'Yes']
    plt.pie(copy4.churn.value_counts(), labels = ['No Churn', 'Churn'], colors="gr", autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Tech Support - Churn %', fontsize = 25, fontweight = 'bold')
    
    plt.subplot(267)
    copy7 = copy[copy.online_security == 'No']
    plt.pie(copy7.churn.value_counts(), labels = ['No Churn', 'Churn'], colors="gr", autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('No Online Security - Churn %', fontsize = 25, fontweight = 'bold')
    
    plt.subplot(268)
    copy8 = copy[copy.online_backup == 'No']
    plt.pie(copy8.churn.value_counts(), labels = ['No Churn', 'Churn'], colors="gr", autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('No Online Backup - Churn %', fontsize = 25, fontweight = 'bold')
    
    plt.subplot(269)
    copy9 = copy[copy.device_protection == 'No']
    plt.pie(copy9.churn.value_counts(), labels = ['No Churn', 'Churn'], colors="gr", autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('No Device Protection - Churn %', fontsize = 25, fontweight = 'bold')
    
    plt.subplot(2, 6, 10)
    copy10 = copy[copy.tech_support == 'No']
    plt.pie(copy10.churn.value_counts(), labels = ['No Churn', 'Churn'], colors="gr", autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('No Tech Support - Churn %', fontsize = 25, fontweight = 'bold')
    

    
    plt.tight_layout()

def plot_contracts(df):
    
    copy = df.copy()
    
    plt.figure(figsize = (30, 10))
    
    plt.subplot(131)
    plt.pie(copy.contract_type.value_counts(), labels = ['Month-to-month', '1 year', '2 year'], autopct = '%.1f%%', textprops = {'fontweight':'bold', 'fontsize': 20})
    plt.title('Customer Contract Composition', fontweight = 'bold', fontsize = 30)
    
    plt.subplot(132)
    plt.title('Churn % by Contract Type', fontsize = 30, fontweight = 'bold')
    copy = copy.groupby('contract_type')['churn'].value_counts().to_frame()
    copy = copy.rename({'churn':'pct_total'}, axis = 1).reset_index()
    copy['pct_total'] = (copy['pct_total']/len(df)) * 100
    a = sns.barplot('contract_type', y = 'pct_total', hue = 'churn', palette=['#008000','#FF0000'], data = copy)
    a.set_title('% Churn - Contract Type', fontsize = 30, fontweight = 'bold')
    a.set(xticklabels = ['Monthly', '1-Year', '2-Year'])
    a.set_xlabel('')
    a.set_ylabel('% of Customers', fontweight = 'bold')
    
def plot_pay_methods(df):
    
    copy = df.copy()
    
    plt.figure(figsize = (30, 10))
    
    plt.subplot(131)
    plt.pie(copy.payment_type.value_counts(), labels = ['Electronic check', 'Mailed check' , 'Bank transfer (automatic)', 'Credit card (automatic)'], autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'}, startangle = -90)
    plt.title('Customer Payment Method Composition', fontsize = 30, fontweight = 'bold')
    
    plt.subplot(132)
    copy = copy.groupby('payment_type')['churn'].value_counts().to_frame()
    copy = copy.rename({'churn':'pct_total'}, axis = 1).reset_index()
    copy['pct_total'] = (copy['pct_total']/len(df))*100
    a = sns.barplot('payment_type', 'pct_total', 'churn', palette=['#008000','#FF0000'], data = copy)
    a.set_title('% Churn - Payment Methods', fontsize = 30, fontweight = 'bold')
    a.set_xlabel('')
    a.set_ylabel('% of Customers', fontsize = 20, fontweight = 'bold')
    a.set_xticklabels(a.get_xticklabels(), rotation = 45)
    
def get_churn_heatmap(df):
    plt.figure(figsize=(8,12))
    churn_heatmap = sns.heatmap(df.corr()[['churn_encoded']].sort_values(by='churn_encoded', ascending=False), vmin=-.5, vmax=.5, annot=True,cmap='flare')
    churn_heatmap.set_title('Features Correlated with Churn')
    
    return churn_heatmap

# Create a function to generate countplots:
def countplot(x, y, df):
    plots = {1 : [111], 2: [121, 122], 3: [131, 132, 133], 4: [221, 222, 223, 224],
         5: [231, 232, 233, 234, 235], 6: [231, 232, 233, 234, 235, 236]}
        
    rows = int(str(plots[len(y)][0])[0])
    columns = int(str(plots[len(y)][0])[1])
    
    plt.figure(figsize=(6*columns, 6*rows))
    
    for i, j in enumerate(y):
        plt.subplot(plots[len(y)][i])
        ax = sns.countplot(x=j, hue=x, data=df, palette=['#008000','#FF0000'], alpha=1, linewidth=0.8, edgecolor="black")
        ax.set_title(j)
        
    return plt.show
    
    plt.tight_layout()   
   
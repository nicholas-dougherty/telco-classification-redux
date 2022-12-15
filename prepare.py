import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def monthly_charges_splits(df) :   
    if df['monthly_charges'] <= 25 :
        return '0-25'
    elif (df['monthly_charges'] > 25) & (df['monthly_charges'] <= 50 ):
        return '26-50'
    elif (df['monthly_charges'] > 50) & (df['monthly_charges'] <= 75 ):
        return '51-75'
    elif (df['monthly_charges'] > 75) & (df['monthly_charges'] <= 100 ):
        return '76-100'
    else:
        return '>100'

def total_charges_splits(df) :   
    if df['total_charges'] <= 2000 :
        return '0-2k'
    elif (df['total_charges'] > 2000) & (df['total_charges'] <= 4000 ):
        return '2k-4k'
    elif (df['total_charges'] > 4000) & (df['total_charges'] <= 6000) :
        return '4k-6k'
    else:
        return '>6k'
    
def tenure_splits(df) :   
    if df['tenure'] <= 6:
        return '1-6'
    elif (df['tenure'] > 6) & (df['tenure'] <= 12 ):
        return '7-12'
    elif (df['tenure'] > 12) & (df['tenure'] <= 18) :
        return '13-18'
    elif df['tenure'] > 18 & (df['tenure'] <= 24) :
        return '19-24'
    else:
        return '>24'



def split_telco_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    # the initial 80/20 split. the test set constitutes 20% of the original df.
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.churn)
    
    # the subsequent 70/30 split. For the remaining 80%, .7 goes to train and .3 to validate
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.churn)
    return train, validate, test

def prep_telco_data(df):
    # Drop duplicate columns
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace=True)
       
    # Drop null values stored as whitespace    
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    
    # Convert to correct datatype
    df['total_charges'] = df.total_charges.astype(float)

    # Convert binary categorical variables to numeric
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    
    # Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(df[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type']], dummy_na=False, \
                              drop_first=False)
    
    # Concatenate dummy dataframe to original 
    df = pd.concat([df, dummy_df], axis=1)
               
    # create categorical columns for these numerical fields
    df['monthlycharges_group'] = df.apply(lambda df:monthly_charges_splits(df), axis = 1)
    df['totalcharges_group'] = df.apply(lambda df:total_charges_splits(df), axis = 1)
    df['tenure_months'] = df.apply(lambda df:tenure_splits(df), axis = 1)
    
    # now I can divvy up these groups of charges and be on my way to scaling. 
    numerical_numbskulls = ['monthlycharges_group','totalcharges_group','tenure_months']
    for col in numerical_numbskulls:
        dummy_df = pd.get_dummies(df[col],
                                  prefix=f'enc_{df[col].name}',
                                  drop_first=False,
                                  dummy_na=False)        
        # add the columns to the dataframe
        df = pd.concat([df, dummy_df], axis=1)
    df = df.drop(columns=numerical_numbskulls)
    
    num_cols = ['monthly_charges', 'total_charges']

    #Scaling Numerical columns
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[num_cols])
    scaled = pd.DataFrame(scaled,columns=num_cols)
    
    #dropping original values merging scaled values for numerical columns
    df = df.drop(columns = num_cols, axis = 1)
    df = df.merge(scaled, left_index=True, right_index=True, how = "left")
    
        # Selective dummy removal since drop_first left unwanted cols and removed favored
    # kept it mostly random as far as selection of what to drop goes. Purposefully
    # excluded no internet service when optional. 
    df = df.drop(columns=['contract_type_One year', 'device_protection_No internet service',
                          'enc_monthlycharges_group_0-25', 'enc_tenure_months_19-24',
                          'enc_totalcharges_group_4k-6k', 'internet_service_type_DSL',
                          'multiple_lines_No phone service','online_backup_No internet service',
                          'online_security_No internet service', 'payment_type_Mailed check',
                          'streaming_movies_No internet service', 'streaming_tv_No internet service',
                          'tech_support_No internet service'
                         ]
                )
                   
    df = df.rename(columns={'internet_service_type_DSL': 'dsl',
                                   'internet_service_type_Fiber optic': 'fiber_optic',
                                   'internet_service_type_None': 'no_internet',
                                   'contract_type_Month-to-month': 'monthly',
                                   'contract_type_Two year': 'two_year_contract',
                                   'payment_type_Bank transfer (automatic)': 'auto_bank_transfer',
                                   'payment_type_Credit card (automatic)': 'auto_credit_card',
                                   'payment_type_Electronic check': 'electronic_check'
                            }
                   )
    
    
    # split the data
    train, validate, test = split_telco_data(df)
    
    return train, validate, test

# That's all for now. any other changes can happen in the explore or preprocess file. 
from env import host, username, password, get_db_url
import os
import pandas as pd 

def get_telco_data(use_cache=True):
# filename = 'telco_churn.csv'
    
    if os.path.exists('telco.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('telco.csv')
    print('Acquiring data from SQL database')
    df = pd.read_sql('''   
                    SELECT * 
                        FROM customers
                        JOIN contract_types USING(contract_type_id)
                        JOIN internet_service_types USING(internet_service_type_id)
                        JOIN payment_types USING(payment_type_id)
                    '''
            , get_db_url('telco_churn'))
    
    df.to_csv('telco.csv', index=False)
    
    return df


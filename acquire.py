from env import host, username, password, get_db_url
import os
import pandas as pd 

def get_telco_data(use_cache=True):
# filename = 'telco_churn.csv'
    """
    This UDF first checks to see whether or not a particular csv exists, and if it does,
    if reads it and returns it as a Dataframe. However, if this check fails, it begins 
    reading the two essential .csvs in someone's local directory, each of which was 
    taken from Codeup's MySQL server (for more information, check the README.md). After reading these 
    dataframes, it performs some necessary extractions, sorting, indexing, ordering, 
    conversions, column creations, de-duplication, and filtration based on unnecessary
    information. Once all of this has run the rounds, it returns a somewhat dirty data
    frame, which will be cleaned via the following prep_data function.
    """
    # If the cached parameter is True, read the csv file on disk in the same folder as this file
    if os.path.exists('telco.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('telco.csv')
    
    # When there's no cached csv, read the following query from Codeup's MySQL database.
    print('Acquiring data from MySQL database')
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


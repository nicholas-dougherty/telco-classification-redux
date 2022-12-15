# Telco Classification Project
 
# Plan -> Acquire -> Prepare -> Explore -> Model & Evaluate -> Deliver


# Objectives:
- Code documentation and elaboration as pertaining to data acquisition & preparation, exploration, and evaluative modeling via Jupyter Notebook. 
- Implement User-Defined Functions for the sake of readability and efficiency.
- Construct an effective machine learning model which can predict customer churn via classification technique.
- Deliver a 5 minute presentation via a notebook walkthrough for an audience of fellow cohorts and data science instructors, acting as notional figureheads at Telco
- Address any questions and concerns as they may arise.

# Business Goals/ The Plan:
- Discover and indicate the primary drivers of churn at this telecommuncations company
- Create an effective machine learning model that will assess churn with high accuracy
- Cleanly and concisely document this process in a notebook, with an aim of reproducibility by anyone interested in following suit. 





### Data Dictionary
---
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
|payment\_type\_id |How a customer pays their bill each month | int64 |
|contract\_type\_id|Which contract type a customer has | int64 |
|internet\_service\_type_id|Type of internet service a customer has | int64 |
|customer\_id|Alpha-numeric ID that identifies each customer| object |
gender|Gender of the customer| object |
senior_citizen|If customer is 65 or older| int64 |
partner|If customer is married| object | 
dependents|If a customer lives with dependents| object |
tenure|The length of a customers relationship with Telco™ measured in months|  int64 |
phone_service|If a customer has phone service| object |
multiple_lines|If a customer has multiple phone lines| object |
online_security|If a customer has online security add-on| object |
online_backup|If a customer has online backups add-on| object |
device_protection|If a customer has a protection plan for Telco™ devices| object |
tech_support|Whether a customer has technical support add-on| object |
streaming_tv|If a customer uses internet to stream tv| object |
streaming_movies|If a customer uses internet to stream movies| object |
paperless_billing|If a customer is enrolled in paperless billing| object |
monthly_charges|The amount a customer pays each month| object |
total_charges|The total amount a customer has paid for Telco™ services| object |
|internet\_service\_type|Type of internet service a customer has| object |
|contract_type|The type of contract a customer has| object |
|payment_type|How a customer pays their bill| object |

| Target | Definition | Data Type |
| ----- | ----- | ----- |
|churn|Indicates whether a customer has terminated service| object |


# Key Findings
- Statistical tests demonstrate that low-tenure, monthly payments, paying via electronic checks, and not opting for optional services as an internet user all contribute to churn in significant dimensions. 


# Model
- Decision Tree (criterion = 'entropy', max_depth = 6, min_samples_leaf = 1, min_samples_split = 30)
- Random Forest Classifier (max_depth = 5, random_state = 123)
- K-Nearest Neighbors(n_neighbors = 7, weights = 'uniform')
- 

# Recommendations:
- Offer incentives targeting churned customers with common features.
- Improve model performance by focusing specifically focusing on internet users.
- Offer discounted services the first 6 months to improve tenure or offer shorter contract periods, i.e. quarterly.


# How to recreate this project:
- You'll need:
 - Python
 - SQL or Kaggle to access the customer information

 - Libraries:
 - pandas 
 - numpy 
 - matplotlib/seaborn
 - sklearn 
 - stats 
# classification-redux
# classification-redux

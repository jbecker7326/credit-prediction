import requests

url = 'http://localhost:9696/predict'

client_id = 'asdfghjkl728'
client = {
    "checking_status":                             1,
    'duration':                                   18,
    'credit_history':                              2,
    'credit_amount':                            3190,
    'savings_status':                              1,
    'employment':                                  2,
    'installment_commitment':                      2,
    'personal_status':                             0,
    'other_parties':                               1,
    'residence_since':                             2,
    'age':                                        24,
    'other_payment_plans':                         1,
    'housing':                                     2,
    'existing_credits':                            1,
    'job':                                         2,
    'num_dependents':                              1,
    'own_telephone':                               0,
    'foreign_worker':                              1,
    'property_magnitude_car':                      0,
    'property_magnitude_life insurance':           0,
    'property_magnitude_no known property':        0,
    'property_magnitude_real estate':              1   
}

response = requests.post(url, json=client).json()

if response['credit_default'] == True:
    print('sending credit loan to %s' % client_id)
else:
    print('not sending credit loan to %s' % client_id)
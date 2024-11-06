#!/usr/bin/env python
# coding: utf-8

# In[1]:


#task1
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

data = {
    'Weather': ['Sunny', 'sunny', 'overcast', 'Rainy', 'Rainy', 'Rainy',
                'overcast', 'Sunny', 'sunny', 'Rainy', 'Sunny',
                'Overcast', 'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool',
                    'Cool', 'Mild', 'Cool', 'Mild', 'Mild',
                    'Mild', 'Hot', 'Mild'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No',
             'Yes','No','Yes','Yes','Yes','Yes','Yes','No']
}
df = pd.DataFrame(data)
le = preprocessing.LabelEncoder()
df['Weather'] = le.fit_transform(df['Weather'].str.lower())  
df['Temperature'] = le.fit_transform(df['Temperature'])
df['Play'] = le.fit_transform(df['Play'])
features = list(zip(df['Weather'], df['Temperature']))
features_train, features_test, label_train, label_test = train_test_split(features, df['Play'], test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(features_train, label_train)
predicted = model.predict(features_test)
print("Prediction:", predicted)
cm = confusion_matrix(label_test, predicted)
print("Confusion Matrix:\n", cm)
accuracy = accuracy_score(label_test, predicted)
print("Accuracy:", accuracy)
TP = cm[1][1]  
TN = cm[0][0]  
FP = cm[0][1] 
FN = cm[1][0]  

if (TP + TN + FP + FN) > 0:
    calculated_accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(f"Calculated Accuracy: {calculated_accuracy:.2f} or {calculated_accuracy * 100:.2f}%")
else:
    print("No predictions were made.")


# In[4]:


#task2
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
data = {
    'Age': ['youth', 'medium', 'senior', 'youth', 'youth', 'senior', 'medium', 'youth'],
    'Income': ['high', 'low', 'high', 'medium', 'low', 'low', 'medium', 'medium'],
    'Student': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'yes', 'no'],
    'Credit Rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'fair', 'fair'],
    'Buy': ['no', 'yes', 'yes', 'yes', 'no', 'no', 'yes', 'no']
}

df = pd.DataFrame(data)

label_encoder = LabelEncoder()

df['Age'] = label_encoder.fit_transform(df['Age'])
df['Income'] = label_encoder.fit_transform(df['Income'])
df['Student'] = label_encoder.fit_transform(df['Student'])
df['Credit Rating'] = label_encoder.fit_transform(df['Credit Rating'])
df['Buy'] = label_encoder.fit_transform(df['Buy'])

print("Encoded DataFrame:\n", df)
X = df[['Age', 'Income', 'Student', 'Credit Rating']]  # Features
y = df['Buy']  # Target (Buy)

nb = GaussianNB()

nb.fit(X, y)

new_instance = pd.DataFrame({
    'Age': [label_encoder.transform(['youth'])[0]],
    'Income': [label_encoder.transform(['medium'])[0]],
    'Student': [label_encoder.transform(['yes'])[0]],
    'Credit Rating': [label_encoder.transform(['fair'])[0]]
})

prediction = nb.predict(new_instance)

predicted_class = label_encoder.inverse_transform(prediction)

print(f"The prediction for the new instance (youth/medium/yes/fair) is: {predicted_class[0]}")


# In[5]:


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
data = {
    'Age': ['Youth', 'Medium', 'Senior', 'Youth', 'Senior', 'Medium', 'Youth', 'Senior'],
    'Income': ['High', 'Low', 'High', 'Medium', 'Low', 'High', 'Low', 'Medium'],
    'Student': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
    'Credit Rating': ['Fair', 'Excellent', 'Fair', 'Fair', 'Excellent', 'Excellent', 'Fair', 'Fair'],
    'Buy': ['No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)
label_encoder = LabelEncoder()

df['Age'] = label_encoder.fit_transform(df['Age'])
df['Income'] = label_encoder.fit_transform(df['Income'])
df['Student'] = label_encoder.fit_transform(df['Student'])
df['Credit Rating'] = label_encoder.fit_transform(df['Credit Rating'])

df['Buy'] = label_encoder.fit_transform(df['Buy'])
print("Encoded DataFrame:\n", df)
X = df[['Age', 'Income', 'Student', 'Credit Rating']]  # Features
y = df['Buy']  # Target variable (Buy)

nb = GaussianNB()

nb.fit(X, y)

new_data = pd.DataFrame({
    'Age': [label_encoder.transform(['Youth'])[0]],
    'Income': [label_encoder.transform(['Medium'])[0]],
    'Student': [label_encoder.transform(['Yes'])[0]],
    'Credit Rating': [label_encoder.transform(['Fair'])[0]]
})

prediction = nb.predict(new_data)

predicted_class = label_encoder.inverse_transform(prediction)
print(f"The prediction for the new instance (Youth, Medium Income, Yes, Fair Credit Rating) is: {predicted_class[0]}")


# In[ ]:





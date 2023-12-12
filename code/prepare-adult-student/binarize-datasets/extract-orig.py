import pandas as pd
from sklearn import preprocessing

# STUDENT

student = pd.read_csv("student-por.csv")

stu = student[['address','Pstatus','schoolsup','famsup','paid','nursery','higher','internet','romantic']].copy()

student2 = pd.get_dummies(stu,columns = ['address','Pstatus','schoolsup','famsup','paid','nursery','higher','internet','romantic'])

df_data = student2.loc[:,student2.columns != 'G3']
df_target = student['G3']

x_train, y_train = df_data.astype(int), df_target

# now save the .csv files
x_train.to_csv('orig/Student_orig_train_data.csv', index=False)
y_train.to_csv('orig/Student_orig_train_labels.csv', index=False)


# ADULT

df = pd.read_csv('adult.csv')

d = df[['capital_gain','capital_loss','type_employer','education','marital','occupation','relationship','race','sex', 'country']].copy()

df['income'] = df['income'].str.replace('<=50K', '0')
df['income'] = df['income'].str.replace('>50K', '1')
df['income'] = df['income'].astype(int)

df2 = pd.get_dummies(d,columns = ['capital_gain','capital_loss','type_employer','education','marital','occupation','relationship','race','sex', 'country'])

df_data = df2.loc[:,df2.columns != 'income']
df_target = df['income']

x_train, y_train = df_data.astype(int), df_target

# now save the .csv files
x_train.to_csv('orig/Adult_orig_train_data.csv', index=False)
y_train.to_csv('orig/Adult_orig_train_labels.csv', index=False)

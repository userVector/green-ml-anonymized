import pandas as pd
from sklearn import preprocessing

k = 3

# STUDENT

# load data
student = pd.read_csv("output/student_" + str(k) + ".csv", delimiter=';')

# filter supressed columns
student = student.loc[:, student.nunique() != 1]

# filter supressed rows
student = student.loc[student['schoolsup'] != "*"]

# convert student grade to pass or fail
student.loc[student['G3'] < 10, 'G3'] = 0
student.loc[student['G3'] > 9, 'G3'] = 1
student['G3'] = student['G3'].astype(int)

if k == 3:
	student2 = pd.get_dummies(student,columns = ['address','Pstatus','schoolsup','famsup','paid','nursery','higher','internet','romantic'])

# split training data and label
df_data = student2.loc[:,student2.columns != 'G3']
df_target = student['G3']

x_train, y_train = df_data.astype(int), df_target

# now save the .csv files
x_train.to_csv('output/processed/Student_' + str(k) + '_train_data.csv', index=False)
y_train.to_csv('output/processed/Student_' + str(k) + '_train_labels.csv', index=False)


# ADULT

# load data
df = pd.read_csv('output/adult_' + str(k) + '.csv', delimiter=';')

# filter suppressed columns (*)
df = df.loc[:, df.nunique() != 1]

# filter suppressed rows (*)
df = df.loc[df['type_employer'] != "*"]

# make binary labels for income column
df['income'] = df['income'].str.replace('<=50K', '0')
df['income'] = df['income'].str.replace('>50K', '1')
df['income'] = df['income'].astype(int)

# use get_dummies for categorical columns
if k == 3:
    df2 = pd.get_dummies(df,columns = ['capital_gain','capital_loss','type_employer','education','marital','occupation','relationship','race','sex', 'country'])

df_data = df2.loc[:,df2.columns != 'income']
df_target = df['income']

x_train, y_train = df_data.astype(int), df_target

# now save the .csv files
x_train.to_csv('output/processed/Adult_' + str(k) + '_train_data.csv', index=False)
y_train.to_csv('output/processed/Adult_' + str(k) + '_train_labels.csv', index=False)

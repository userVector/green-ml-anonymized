import pandas as pd
import random
import os
import numpy as np

def sample_dataframe(csv_file, csv_y, sample_method, num_columns, my_list, num_rows, rows_replace):
    # Read the CSV file into a Pandas DataFrame
    data = pd.read_csv(csv_file)
    
    # Set a seed for reproducibility
    np.random.seed(42)
    
    # Perform sampling based on the specified method
    if sample_method == 'with_replacement':
        starting_letters = random.choices(my_list, k=num_columns)
    elif sample_method == 'without_replacement':
        starting_letters = random.sample(my_list, num_columns)

    # Find columns that start with the specified letters based on sampled columns
    columns_to_include = []
    for start in starting_letters:
        columns_to_include.extend([col for col in data.columns if col.startswith(start)])

    # Create a new DataFrame with columns starting with the specified letters
    sampled_data = data[columns_to_include]
    
    df2 = pd.read_csv(csv_y)
    # Add column from df2 to df1 while preserving its name
    sampled_data = sampled_data.join(df2[df2.columns.intersection(df2.columns.difference(sampled_data.columns))])
    
    sampled_rows = sampled_data.sample(n=num_rows, replace=rows_replace)  # n is the number of rows to sample
    
    column_name = df2.columns[0]
    column_to_move = sampled_rows.pop(df2.columns[0])  # Remove and return the specified column
    sampled_y = pd.DataFrame({df2.columns[0]: column_to_move})  # Create a new DataFrame with the removed column

    return sampled_rows, sampled_y
	

# Values for the loops
values1 = [10, 15, 20]
values2 = [1000, 5000, 10000]


# student

if not os.path.exists('student_data'):
    os.makedirs('student_data')
os.chdir('student_data')

file_path = 'C:/Users/Vit/Desktop/gl-paper/prepare-testing/student_data/Student_orig_train_data.csv'
y_path = 'C:/Users/Vit/Desktop/gl-paper/prepare-testing/student_data/Student_orig_train_labels.csv'
k_file_path = 'C:/Users/Vit/Desktop/gl-paper/prepare-testing/student_data/Student_3_train_data.csv'
k_y_path = 'C:/Users/Vit/Desktop/gl-paper/prepare-testing/student_data/Student_3_train_labels.csv'

starting_letters = ['address','Pstatus','schoolsup','famsup','paid','nursery','higher','internet','romantic']  # Letters corresponding to column starts

if not os.path.exists('original_training_data'):
    os.makedirs('original_training_data')
os.chdir('original_training_data')

for val2 in values2:
    for val1 in values1:
        folder_name = f"{val2}r_{val1}f"
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        for i in range(2):
            sampled_df, sampled_y = sample_dataframe(file_path, y_path, 'with_replacement', val1, starting_letters, val2, True)
            sampled_df.to_csv(folder_name+'/X_student'+str(i+1)+'.csv', index=False)
            sampled_y.to_csv(folder_name+'/y_student'+str(i+1)+'.csv', index=False)

os.chdir('..')
if not os.path.exists('anonymized_training_data'):
    os.makedirs('anonymized_training_data')
os.chdir('anonymized_training_data')

for val2 in values2:
    for val1 in values1:
        folder_name = f"{val2}r_{val1}f"
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        for i in range(2):
            sampled_df, sampled_y = sample_dataframe(k_file_path, k_y_path, 'with_replacement', val1, starting_letters, val2, True)
            sampled_df.to_csv(folder_name+'/X_student_anon'+str(i+1)+'.csv', index=False)
            sampled_y.to_csv(folder_name+'/y_student_anon'+str(i+1)+'.csv', index=False)


# adult

os.chdir('..')
os.chdir('..')
if not os.path.exists('adult_data'):
    os.makedirs('adult_data')
os.chdir('adult_data')

file_path = 'C:/Users/Vit/Desktop/gl-paper/prepare-testing/adult_data/Adult_orig_train_data.csv'
y_path = 'C:/Users/Vit/Desktop/gl-paper/prepare-testing/adult_data/Adult_orig_train_labels.csv'
k_file_path = 'C:/Users/Vit/Desktop/gl-paper/prepare-testing/adult_data/Adult_3_train_data.csv'
k_y_path = 'C:/Users/Vit/Desktop/gl-paper/prepare-testing/adult_data/Adult_3_train_labels.csv'

starting_letters = ['capital_gain','capital_loss','type_employer','education','marital','occupation','relationship','race','sex', 'country']  # Letters corresponding to column starts

if not os.path.exists('original_training_data'):
    os.makedirs('original_training_data')
os.chdir('original_training_data')

for val2 in values2:
    for val1 in values1:
        folder_name = f"{val2}r_{val1}f"
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        for i in range(2):
            sampled_df, sampled_y = sample_dataframe(file_path, y_path, 'with_replacement', val1, starting_letters, val2, False)
            sampled_df.to_csv(folder_name+'/X_adult'+str(i+1)+'.csv', index=False)
            sampled_y.to_csv(folder_name+'/y_adult'+str(i+1)+'.csv', index=False)

os.chdir('..')
if not os.path.exists('anonymized_training_data'):
    os.makedirs('anonymized_training_data')
os.chdir('anonymized_training_data')

for val2 in values2:
    for val1 in values1:
        folder_name = f"{val2}r_{val1}f"
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        for i in range(2):
            sampled_df, sampled_y = sample_dataframe(k_file_path, k_y_path, 'with_replacement', val1, starting_letters, val2, False)
            sampled_df.to_csv(folder_name+'/X_adult_anon'+str(i+1)+'.csv', index=False)
            sampled_y.to_csv(folder_name+'/y_adult_anon'+str(i+1)+'.csv', index=False)

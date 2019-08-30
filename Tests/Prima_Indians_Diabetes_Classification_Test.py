#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import pandas as pd

from sklearn.model_selection import train_test_split

from AutoAIAlgorithm.ParticleSwarmOptimization import PSO

from sklearn.metrics import f1_score


dataset_prima_indians_name = "..\\Datasets\\Classification\\Prima Indians Diabetes\\diabetes_prima_indians.csv"
label_column_name = "Outcome"

#--- MAIN ---------------------------------------------------------------------+
def main():
    dataset = pd.read_csv(dataset_prima_indians_name)

    # calculate the mean of each column which will be used 
    # to eliminate Na values (missing values)
    mean_Glucose = dataset['Glucose'].mean(skipna=True)
    mean_BloodPressure = dataset['BloodPressure'].mean(skipna=True)
    mean_SkinTickness = dataset['SkinThickness'].mean(skipna=True)
    mean_Insulin = dataset['Insulin'].mean(skipna=True)
    mean_BMI = dataset['BMI'].mean(skipna=True)
    mean_DiabetesPedigreeFunction = dataset['DiabetesPedigreeFunction'].mean(skipna=True)
    
    # replace missing values this the mean
    dataset['Glucose'] = dataset['Glucose'].replace(0, mean_Glucose)
    dataset['BloodPressure'] = dataset['BloodPressure'].replace(0, mean_BloodPressure)
    dataset['SkinThickness'] = dataset['SkinThickness'].replace(0, mean_SkinTickness)
    dataset['Insulin'] = dataset['Insulin'].replace(0, mean_Insulin)
    dataset['BMI'] = dataset['BMI'].replace(0, mean_BMI)
    dataset['DiabetesPedigreeFunction'] = dataset['DiabetesPedigreeFunction'].replace(0, mean_DiabetesPedigreeFunction)

    x_tr, test = train_test_split(dataset, test_size=0.2, shuffle=False, random_state=41)
    train, valid = train_test_split(x_tr, test_size=0.25, shuffle=False, random_state=41)
    
    xtrain, ytrain = train.drop(label_column_name, axis=1), train[label_column_name]
    xvalid, yvalid = valid.drop(label_column_name, axis=1), valid[label_column_name]
    
    num_particles=20
    num_iterations=50
    
    pso = PSO(particle_count=num_particles, evaluation_metric=f1_score)
    
    best_metric, best_model = pso.fit(X_train=xtrain,
                                      X_test=xvalid,
                                      Y_train=ytrain,
                                      Y_test=yvalid,
                                      maxiter=num_iterations,
                                      verbose=True,
                                      max_distance=0.05)
            
    print("BEST")
    print(best_metric)
    print(best_model)


if __name__ == "__main__":
    main()
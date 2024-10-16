from sklearn.model_selection import train_test_split
import dataset
import pandas as pd



def load_data():
    train_data={
        'feature1':[1,2,3,4,5],
        'feature2':[10,20,30,40,50],
        'label':[0,1,0,1,0]

    }
    dataset=load_data()
df=pd.DataFrame(dataset['train'])
train_df, val_df = train_test_split(df,test_size=0.2, random_state=42)
print(f"Traingin set size :{len(train_df)}, Validationn Set Size: {len(val_df)}")

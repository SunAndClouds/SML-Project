from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
    
def create_numpy_dataset(seed=0, train_size=0.8):
    # Load the dataset
    file_path = 'training_data.csv'
    data = pd.read_csv(file_path)
    data['increase_stock_binary'] = data['increase_stock'].apply(
        lambda x: 1 if x == 'high_bike_demand' else 0
    )

    X = data.to_numpy()
    X = np.concatenate([X[:,:-2], X[:,-1:]], axis=-1) 

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    y = X[:,-1].astype(np.float32)
    X = X[:,:-2].astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=train_size, 
        random_state=seed, 
        shuffle=True
    )
    
    return X_train, X_test, y_train, y_test


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Generate test dataset
def generate_seq_single_preFailure(df,failure_time,features_columns,timewindow_for_use,window_len,stride):
  '''
  Generate the test data set using one machine failure time.
  '''
  
  X = np.empty((1,1,window_len*len(features_columns)), float)
  Y=np.empty((1), float)
  
  windows_start=failure_time-pd.Timedelta(seconds=60*timewindow_for_use[0]) #  mins before the failure time
  windows_end=failure_time-pd.Timedelta(seconds=60*timewindow_for_use[1]) #  mins before the failure time
  df_prefailure_single_window_feature=df.loc[windows_start:windows_end,features_columns]
  df_prefailure_single_window_target=df.loc[windows_start:windows_end,'alarm']
    
  data=df_prefailure_single_window_feature.to_numpy().tolist()
  targets=df_prefailure_single_window_target.tolist()

  data_gen=tf.keras.preprocessing.sequence.TimeseriesGenerator(data, targets, window_len,stride=stride,sampling_rate=1,batch_size=1,shuffle=False) # for ploting, do not shuffle the data
  
  for i in range(len(data_gen)):
    x, y = data_gen[i]
    x=np.transpose(x).flatten()
    x=x.reshape((1,1,len(x)))
    X=np.append(X,x,axis=0)
    Y=np.append(Y,y/2,axis=0)

  return X,Y

def inference():
    # Load data
    df_test = pd.read_csv('df_test.csv', index_col='timestamp', parse_dates=True)
    Df_test=df_test.copy()

    df_train = pd.read_csv('df_train.csv', index_col='timestamp', parse_dates=True)
    Df_train=df_train.copy()
    
    # preprocess
    # PCA
    sensor_names=Df_test.columns[:-2]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(Df_train[sensor_names]) # use this scaler for test and validation data set

    test_scaled=scaler.transform(Df_test[sensor_names])
    train_scaled=scaler.transform(Df_train[sensor_names])
    pca=PCA(n_components=8).fit(train_scaled) # use this pca for test and validation data set
    df_test_pca=pd.DataFrame(pca.transform(test_scaled))

    pcs = ['pc'+str(i+1) for i in range(8)]
    df_test_pca.columns = pcs
    df_test_pca['machine_status']=Df_test['machine_status'].values
    df_test_pca['alarm']=Df_test['alarm'].values
    df_test_pca.index=Df_test.index
    df_test=df_test_pca[['pc1','pc2','pc3','pc4','machine_status','alarm']]
    # Return X, y
    failure_time=df_test[df_test['machine_status']==1].index[0]
    features_columns=df_test.columns.tolist()[:-2]
    timewindow_for_use=(60*60,5) # 6h-5min
    window_len=20
    stride=1
    X_test,y_test=generate_seq_single_preFailure(df_test,failure_time,features_columns,timewindow_for_use,window_len,stride)

    id_keep= np.where((y_test == 0) | (y_test ==1))
    y_test=y_test[id_keep]
    X_test=X_test[id_keep][:,:]
    # X_test.shape, y_test.shape
    
    # Run model
    model_1 = tf.keras.models.load_model("my_model.h5") 
    
    # evaluation1: loss and accuracy
    loss,accuracy=model_1.evaluate(X_test,y_test)
    print(f'Model loss on the test set: {loss:.4f}')
    print(f'Model accuracy on the test set: {(accuracy*100):.2f}%')
    
    failure_time=df_test[df_test['machine_status']==1].index[0]
    print('failure time:',failure_time)
    windows_start=failure_time-pd.Timedelta(seconds=60*timewindow_for_use[0])
    windows_end=failure_time-pd.Timedelta(seconds=60*timewindow_for_use[1])
    df_preFailure=df_test.loc[windows_start:windows_end,:]

    y_test_preds_1=model_1.predict(X_test).flatten()
    y_test_preds_1=tf.round(y_test_preds_1)
    df_preFailure['alarm']=np.append(np.zeros(window_len),(y_test_preds_1))

    df_preFailure[['pc1','pc2','pc3','pc4']].plot(c="grey",figsize=(18,2),legend=None)
    df_preFailure[df_preFailure['alarm']==1].index.to_list()

    # vertical lines
    for xc in df_preFailure[df_preFailure['alarm']==1].index.to_list():
        plt.axvline(x=xc,c='red')

    df_test['machine_status'].plot()
    plt.xlim([windows_start,windows_end+pd.Timedelta(seconds=60*200)])

    plt.title("60h ahead of the failure")
    plt.show()
    
if __name__ == '__main__':
    inference()
    
    
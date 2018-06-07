#!/usr/bin/python3
# author: zhaofeng-shu33
# description: collaborative filtering
import pdb
import pandas as pd
import numpy as np
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.sparse as sp
from scipy.sparse.linalg import svds
def rmse(prediction, ground_truth):
    valid_index = np.where(np.logical_and(ground_truth!=0,~np.isnan(prediction)))
    prediction_filtered = prediction[valid_index].flatten()
    ground_truth_filtered = ground_truth[valid_index].flatten()
    return sqrt(mean_squared_error(prediction_filtered, ground_truth_filtered))
TEST_SIZE_PERCENTAGE = 0.05
header = ['user_id','organization_id', 'joint_times']
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
if __name__ == '__main__':
    df = pd.read_csv('./volunteer_recommendv2.csv', sep='\t', names = header)
    
    u_id = {}
    o_id = {}
    pdb.set_trace()    
    for line in df.itertuples(): # line[0] is index
        u_id[line[1]] = 0
        o_id[line[2]] = 0
    n_users = 0
    for key,val in u_id.items():
        u_id[key] = n_users
        n_users += 1
    n_organizations = 0
    for key,val in o_id.items():
        o_id[key] = n_organizations
        n_organizations += 1
    u_id_list = np.zeros(df.shape[0], dtype = int)
    o_id_list = np.zeros(df.shape[0], dtype = int)
    for line in df.itertuples(): # line[0] is index
        u_id_list[line[0]] = u_id[line[1]]        
        o_id_list[line[0]] = o_id[line[2]]
    df = df.join(pd.DataFrame({'u_id' : u_id_list, 'o_id' : o_id_list}))
    
    train_data, test_data = cv.train_test_split(df, test_size=TEST_SIZE_PERCENTAGE)
    #Create two user-item matrices, one for training and another for testing
    train_data_matrix = np.zeros((n_users, n_organizations))
    for line in train_data.itertuples(): # row: user; column: organization
        train_data_matrix[line.u_id, line.o_id] = line.joint_times

    test_data_matrix = np.zeros((n_users, n_organizations))
    for line in test_data.itertuples():
        test_data_matrix[line.u_id, line.o_id] = line.joint_times
    user_similarity = cosine_similarity(train_data_matrix) # possibly produces nan value
    item_similarity = cosine_similarity(train_data_matrix.T)
    user_prediction = predict(train_data_matrix, user_similarity, type='user')   
    item_prediction = predict(train_data_matrix, item_similarity, type='item')
    print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
    print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))  
    # model based
    #get SVD components from train matrix. Choose k.
    u, s, vt = svds(train_data_matrix, k = 20)
    s_diag_matrix=np.diag(s)
    X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
    print('User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix)))
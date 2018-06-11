#!/usr/bin/python3
# author: zhaofeng-shu33
# description: collaborative filtering
import pdb
import pandas as pd
import numpy as np
import logging # logging prediction result to file
from xml.etree import ElementTree as ET
TEST_SIZE_PERCENTAGE = 0.05
org = ['清华大学精仪系紫荆志愿者支队', '清华大学建筑学院紫荆志愿者支队', '清华大学生物系紫荆志愿者支队', '清华大学\
机械系紫荆志愿者支队', '清华大学环境系紫荆志愿者支队', '清华大学热能系紫荆志愿者支队', '清华大学爱心公益协会\
', '清华大学化学系紫荆志愿者支队', '清华大学学生教育扶贫公益协会', '清华大学红十字会', '北京大学生命科学学院\
青年志愿者协会', '清华大学航天航空学院紫荆志愿者支队', '北京大学青年志愿者协会', '清华大学领航志愿服务团', '\
北京大学红十字会学生分会', '清华大学学生绿色协会', '清华大学自动化系紫荆志愿者支队', '清华大学汽车系紫荆志愿\
者支队', '清华大学义务讲解支队', '清华大学经济管理学院紫荆志愿者支队', '清华大学土木系紫荆志愿者支队', '清华\
大学唐仲英爱心社', '清华大学电机系紫荆志愿者支队', '清华大学水利系紫荆志愿者支队', '清华大学化工系紫荆志愿者\
支队', '清华大学计算机系紫荆志愿者支队', '清华大学材料系紫荆志愿者支队', '清华大学电子系紫荆志愿者支队', '北\
京大学元培学院青年志愿者协会', '清华大学美术学院紫荆志愿者支队', '清华大学学生书脊支教团', '北京大学法律援助\
协会', '清华大学工业工程系紫荆志愿者支队', '清华大学工物系紫荆志愿者支队', '清华大学学习发展中心', '北京大学\
化学与分子工程学院青年志愿者协会', '北京大学心理学系青协', '清华大学法学院紫荆志愿者支队', '清华大学手语社',\
 '清华大学医学院紫荆志愿者支队', '北京大学经济学院青年志愿者协会', '清华大学粉刷匠工作室“优化校园，粉刷梦想\
”志愿服务队', '清华大学物理系紫荆志愿者支队', '北京大学光华管理学院青年志愿者协会', '清华大学国际志愿服务队\
', '北京大学流浪猫关爱协会', '清华大学软件学院紫荆志愿者支队', '北京大学百周年纪念讲堂学生志愿者协会', '清华\
大学数学系紫荆志愿者支队', '北京大学政府管理学院团委青年志愿者协会', '清华大学人文学院志愿服务团', '清华大学\
社科学院紫荆志愿者支队', '北京大学爱心社', '北京大学环境科学与工程学院团委青年志愿者协会', '北京大学信息管理\
系青年志愿者协会', '北京大学软微学院青年志愿者协会', '北京大学城市与环境学院青年志愿者协会', '北京大学外国语学院青年志愿者协会', '北京大学物\
理学院青年志愿者协会', '北京大学地球与空间科学学院青年志愿者协会', '北京大学艺术学院青年志愿者协会', '北京大\
学校史馆志愿讲解队', '清华大学学生关爱留守儿童协会', '北京大学信息科学技术学院青年志愿者协会', '北京大学社会\
学系青年志愿者协会', '清华大学新闻与传播学院志愿中心']
header = ['user_id','organization_id', 'joint_times']
volunteer_header = ['volunteer_id', 'volunteer_name']
def predict(ratings, similarity):
    pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred.reshape(max(pred.shape))
if __name__ == '__main__':
    df = pd.read_csv('./volunteer_recommendv2.csv', sep = '\t', names = header)
    volunteer_table = pd.read_csv('./volunteer_info.csv', sep = '\t', names = volunteer_header)
    v_id_name =  {} # dict, mapping volunteer_id to volunteer_name
    for line in volunteer_table.itertuples():
        v_id_name[line.volunteer_id] = line.volunteer_name
    # get the total number of volunteers and organizations
    u_id = {} # dict, mapping user_id to user_matrix_id
    o_id = {} # dict, mapping organization_id to org_matrix_id
    for line in df.itertuples(): # line[0] is index
        u_id[line.user_id] = 0
        o_id[line.organization_id] = 0
    n_users = 0
    for key,val in u_id.items():
        u_id[key] = n_users
        n_users += 1
    n_organizations = 0
    for key,val in o_id.items():
        o_id[key] = n_organizations
        n_organizations += 1

    u_id_list = np.zeros(df.shape[0], dtype = int) # list, u_id_list[index], get the user_id when user_matrix_id = index
    for line in df.itertuples(): # line[0] is index
        u_id_list[line.Index] = line.user_id
    
    train_data = pd.read_csv('./train_data.csv', sep=',')
    test_data = pd.read_csv('./test_data.csv', sep = ',')
    
    #Create two user-item matrices, one for training and another for testing
    train_data_matrix = np.zeros((n_users, n_organizations))
    for line in train_data.itertuples(): # row: user; column: organization
        train_data_matrix[line.u_id, line.o_id] = line.joint_times

    test_data_dict = {}
    for line in test_data.itertuples():
        test_data_dict[line.u_id] = []
    for line in test_data.itertuples():
        test_data_dict[line.u_id].append(line.o_id)

    item_similarity = np.load('item_similarity_partial.npx')
    # use acc to measure (top 3)
    num_of_predict_total = 0
    num_of_predict_true = 0
    root = ET.Element('tbody')
    # pdb.set_trace()
    for key,org_id_list in test_data_dict.items():
        # get the activity he already joined, 
        # train_data_matrix[key,:]
        # if non, pass
        num_activity_joined= np.sum(train_data_matrix[key,:]>0)
        if(num_activity_joined == 0):
            continue
        item_prediction_tmp = predict(train_data_matrix[key,:], item_similarity)
        item_prediction = []
        for i in range(item_prediction_tmp.shape[0]):
            item_prediction.append([i,item_prediction_tmp[i]])
        item_prediction.sort(key=lambda x:x[1], reverse = True)        # sort as descending
        item_prediction_index = [ i[0] for i in item_prediction]
        item_prediction_best = []
        for i in range(num_activity_joined + 3): # select three more
            item_prediction_best.append(item_prediction[i][0])
        one_tr = ET.SubElement(root, 'tr')
        one_td = ET.SubElement(one_tr, 'td')
        one_td.text = v_id_name[u_id_list[key]] # 'volunteer_name'
        joined = np.where(train_data_matrix[key,:]>0)
        activity_joined = []
        for i in joined[0]:
            activity_joined.append(org[i])
        one_td = ET.SubElement(one_tr, 'td')            
        one_td.text = '<br/>'.join(activity_joined)

        activity_will_join = []        
        for org_id in org_id_list:
            num_of_predict_total += 1
            activity_will_join.append(org[org_id])
            if(item_prediction_best.count(org_id)>0):
                num_of_predict_true += 1
        one_td = ET.SubElement(one_tr, 'td')
        one_td.text = '<br/>'.join(activity_will_join)        

        activity_prediction = []
        for i in item_prediction_best:
            activity_prediction.append(org[i])
        one_td = ET.SubElement(one_tr, 'td')
        one_td.text = '<br/>'.join(activity_prediction)

    print('Total prediction: %d; Total true prediction: %d; error rate: %f'%(num_of_predict_total, num_of_predict_true, 1-num_of_predict_true/num_of_predict_total ))
    template = open('template.html','rb').read()
    content_to_be_replaced = ET.tostring(root, encoding = 'utf-8').replace(b'&lt;',b'<').replace(b'&gt;',b'>')    
    open('item_based_cf_predict_result.html','wb').write(template.replace(b'{content}', content_to_be_replaced))

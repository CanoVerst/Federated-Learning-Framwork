import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import torch

def train_attack_model(shadow_model, shadow_client_loaders, shadow_test_loader, N_class):
    
    min_size = min(len(shadow_client_loaders.sampler), len(shadow_test_loader.sampler))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shadow_model.to(device)
        
    shadow_model.eval()
    ####
    pred_4_mem = torch.zeros([1,N_class])
    pred_4_mem = pred_4_mem.to(device)
    with torch.no_grad():
        data_loader = shadow_client_loaders
        
        for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(device)
                out = shadow_model(data)
                pred_4_mem = torch.cat([pred_4_mem, out])
                if len(pred_4_mem) > min_size:
                    break
    pred_4_mem = pred_4_mem[1:,:]
    pred_4_mem = torch.softmax(pred_4_mem,dim = 1)
    pred_4_mem = pred_4_mem.cpu()
    pred_4_mem = pred_4_mem.sort(axis = 1, descending = True).values[:,:3]
    pred_4_mem = pred_4_mem.detach().numpy()
    
    ####
    pred_4_nonmem = torch.zeros([1,N_class])
    pred_4_nonmem = pred_4_nonmem.to(device)
    with torch.no_grad():
        for batch, (data, target) in enumerate(shadow_test_loader):
            data = data.to(device)
            out = shadow_model(data)
            pred_4_nonmem = torch.cat([pred_4_nonmem, out])
            if len(pred_4_nonmem) > min_size:
                break
    pred_4_nonmem = pred_4_nonmem[1:,:]
    pred_4_nonmem = torch.softmax(pred_4_nonmem,dim = 1)
    pred_4_nonmem = pred_4_nonmem.cpu()
    pred_4_nonmem = pred_4_nonmem.sort(axis = 1, descending = True).values[:,:3]
    pred_4_nonmem = nonmember_filter(pred_4_nonmem, desired_number = min_size)
    pred_4_nonmem = pred_4_nonmem.detach().numpy()
    
    
    #构建MIA 攻击模型 
    att_y = np.hstack((np.ones(pred_4_mem.shape[0]), np.zeros(pred_4_nonmem.shape[0])))
    att_y = att_y.astype(np.int16)
    
    att_X = np.vstack((pred_4_mem, pred_4_nonmem))
    
    
    X_train,X_test, y_train, y_test = train_test_split(att_X, att_y, test_size = 0.1)
    
    attacker = XGBClassifier(n_estimators = 300,
                              n_jobs = -1,
                                max_depth = 30,
                              objective = 'binary:logistic',
                              booster="gbtree",
                              # learning_rate=None,
                               # tree_method = 'gpu_hist',
                               scale_pos_weight = pred_4_nonmem.shape[0]/pred_4_mem.shape[0]
                              )
    

    
    attacker.fit(X_train, y_train)
    print('\n')
    print("MIA Attacker training accuracy")
    print(accuracy_score(y_train, attacker.predict(X_train)))
    print("MIA Attacker testing accuracy")
    print(accuracy_score(y_test, attacker.predict(X_test)))
    
    return attacker

def _attack(target_model, attack_model, client_loaders, test_loader, N_class):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    target_model.to(device)
        
    target_model.eval()
    
    #The predictive output of forgotten user data after passing through the target model.
    mem_X = torch.zeros([1,N_class])
    mem_X = mem_X.to(device)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(client_loaders):
            data = data.to(device)
            out = target_model(data)
            mem_X = torch.cat([mem_X, out])
                    
    mem_X = mem_X[1:,:]
    mem_X = torch.softmax(mem_X,dim = 1)

    mem_X = mem_X.sort(axis = 1, descending = True).values[:,:3]
    mem_X = mem_X.cpu().detach().numpy()
    
    mem_y = np.ones(mem_X.shape[0])
    mem_y = mem_y.astype(np.int16)
    
    N_mem_sample = len(mem_y)
    
    #Test data, predictive output obtained after passing the target model
    test_X = torch.zeros([1, N_class])
    test_X = test_X.to(device)
    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):
            data = data.to(device)
            out = target_model(data)
            test_X = torch.cat([test_X, out])
            
            if(test_X.shape[0] > N_mem_sample):
                break
    test_X = test_X[1:N_mem_sample+1,:]
    test_X = torch.softmax(test_X,dim = 1)
    test_X = test_X.sort(axis = 1, descending = True).values[:,:3]

    test_X = nonmember_filter(test_X, desired_number = N_mem_sample)
    test_X = test_X.cpu().detach().numpy()
    
    test_y = np.zeros(test_X.shape[0])
    test_y = test_y.astype(np.int16)
    
    #The data of the forgotten user passed through the output of the target model, and the data of the test set passed through the output of the target model were spliced together
    #The balanced data set that forms the 50% train 50% test.
    XX = np.vstack((mem_X, test_X))
    YY = np.hstack((mem_y, test_y))
    
    pred_YY = attack_model.predict(XX)
    acc = accuracy_score( YY, pred_YY)
    pre = precision_score(YY, pred_YY, pos_label=1)
    rec = recall_score(YY, pred_YY, pos_label=1)
    print("MIA Attacker accuracy = {:.4f}".format(acc))
    print("MIA Attacker precision = {:.4f}".format(pre))
    print("MIA Attacker recall = {:.4f}".format(rec))
    return (acc, pre, rec)

def nonmember_filter(features, desired_number = None):
    if desired_number is None:
        desired_number = int(len(features) / 2)
    if desired_number == len(features):
        desired_number = int(len(features) / 2)
    selected_features = features.sort(axis = 0).values[:desired_number]
    return selected_features.clone().detach()
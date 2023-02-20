import numpy as np 
from Trainer.Vanilla import NetWork
from utils.helper_functions import read_data



if __name__ == "__main__" :
    
           
    
    
    data = read_data("/home/kuntima/workspace/github/BackPropagation/data/wheat-seeds-binary.csv")
    epoch = 3
    net = {"layers" :{1:{"neurons":7,     "parameters":{}},
                        2:{"neurons":5,         "parameters":{}},
                        3:{"neurons":1,    "parameters":{}},
                        4:{"neurons":1,    "cost":np.zeros((1,1)), "correct_pred":0}}}
    
    net = NetWork(learning_rate=0.01,networkstructure=net)
    for _ in range(epoch) :
        seed = np.random.randint(0, 10000)
        np.random.seed(seed)
        np.random.shuffle(data)
        n = data.shape[0]//2
        #The class values are 1,2 but we want them to be 0,1, so we subtract 1 from the column 'Class'
        # Divide the data in train and test, half for each
        X_train, y_train =  data[:n,:-1], data[:n,-1].reshape(-1, 1) -1
        X_test,  y_test  =  data[n:,:-1], data[n:,-1].reshape(-1, 1) -1 
        net.train(X_train, y_train, iter=100, mode="sigmoid")
        net.test(X_test,y_test)



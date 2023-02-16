import numpy as np 
from Trainer.Vanilla import NetWork




if __name__ == "__main__" :
    input = 13
    output = 3
    net = {"layers" :{1:{"neurons":input,     "parameters":{}},
                      2:{"neurons":11,         "parameters":{}},
                      3:{"neurons":7,         "parameters":{}},
                      4:{"neurons":5,         "parameters":{}},
                      5:{"neurons":3,         "parameters":{}},
                      6:{"neurons":output,    "parameters":{}},
                      7:{"neurons":output,    "cost":np.zeros((output,1))}}}
           
           
           
    net = NetWork(learning_rate=0.5,networkstructure=net)
    X = np.array([ [i] for i in range(input) ])
    y = np.array([[1], [0] , [0] ])
    net.train(X,y, iter=5000)




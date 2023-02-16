import numpy as np 



class NetWork:
    def __init__(self, 
                 learning_rate=0.001,
                 networkstructure = {"layers" :{1:{"neurons":7, "parameters":{}},
                                                2:{"neurons":5, "parameters":{}},
                                                3:{"neurons":3, "parameters":{}},
                                                4:{"neurons":2, "parameters":{}},
                                                5:{"neurons":2, "cost":0}}}) -> None:
        self.learning_rate = learning_rate
        self.networkstructure = networkstructure
        self.L = len(self.networkstructure["layers"])  # Last layer
        self.networkstructure = self.initialize_network(self.networkstructure, 'Xavier')
        self.a0 = 0
    
    def initialize_network(self, networkstructure, mode='Xavier') -> dict:
        # initialize the weights following some distribuition [Xavier, Gaussion, ]
        
        if mode == "Xavier" :
            # initialize the first layer 
            self.networkstructure["layers"][1]["parameters"]["W" ]       = self.xavier_initialization(networkstructure["layers"][1]["neurons"], networkstructure["layers"][1]["neurons"])
            self.networkstructure["layers"][1]["parameters"]["b"]      =  np.ones((networkstructure["layers"][1]["neurons"],1))
            self.networkstructure["layers"][1]["parameters"]["a"]   =  np.ones((networkstructure["layers"][1]["neurons"],1))
            for i in range(1,self.L-1)  :
               self.networkstructure["layers"][i+1]["parameters"]["W" ]  =  self.xavier_initialization(networkstructure["layers"][i]["neurons"], networkstructure["layers"][i+1]["neurons"] )
               self.networkstructure["layers"][i+1]["parameters"]["b"]   =  np.ones((networkstructure["layers"][i+1]["neurons"],1))
               self.networkstructure["layers"][i+1]["parameters"]["a"]   =  np.ones((networkstructure["layers"][i]["neurons"],1))
               
        return networkstructure
    
    def xavier_initialization(self, input, output):
        
        # Gaussian Xavier Initialization
        input_node = input
        output_node= output
        xavier_stddev = np.sqrt(2.0/(input + output))
        return np.random.normal(size = [input_node, output_node], scale=xavier_stddev)
    
    def loss_function(self, y_hat, y) -> float:
 
        return -y*np.log(y_hat) + (1-y)*np.log(1-y_hat)
    
    def update_parameters(self) :
        # updating weigths and bias 
        
        for i in range(1,len(self.networkstructure["layers"]))  :
            self.networkstructure["layers"][i]["parameters"]["W"] += - self.learning_rate*self.networkstructure["layers"][i]["parameters"]["dW"]
            self.networkstructure["layers"][i]["parameters"]["b"] += - self.learning_rate*self.networkstructure["layers"][i]["parameters"]["db"]
            

    def train(self,X, y, iter=1000) :
        
        
        for i in range(iter) :
            self.networkstructure["layers"][self.L]["cost"] = 0
            for j in range(1) :
                # forward propagate
                self.fowardpropagation(X)
            
                # Calculating the cost 
           
                self.networkstructure["layers"][self.L]["cost"] += self.loss_function(self.networkstructure["layers"][self.L-1]["parameters"]["a"], y)
              
                # backpropagation
                self.backpropagation(y)
                
                # update the weigths and bias
                self.update_parameters()
          
            self.networkstructure["layers"][self.L]["cost"] /= X.shape[0]
            print('Iteration: ', iter)
            print("Cost: ", self.networkstructure["layers"][self.L]["cost"])
            print("Predictions :", self.networkstructure["layers"][self.L-1]["parameters"]["a"])
            

   
    def sigmoid(self, Z) -> np.ndarray:
        return 1 / (1 + np.exp(-Z))
    
    def deriv_sigmoid(self, parameters) :
        return parameters["a"]*(1-parameters["a"])
    
    def backpropagation(self,y) :
        # Calcuclating the derivative for the last layer
        self.networkstructure["layers"][self.L-1]["parameters"]["dZ"] =  self.networkstructure["layers"][self.L-1]["parameters"]["a"] - y
        self.networkstructure["layers"][self.L-1]["parameters"]["db"] =  self.networkstructure["layers"][self.L-1]["parameters"]["dZ"]
        self.networkstructure["layers"][self.L-1]["parameters"]["dW"] =  self.networkstructure["layers"][self.L-2]["parameters"]["a"] * self.networkstructure["layers"][self.L-1]["parameters"]["dZ"].T
        
        for l in reversed(range(1, self.L-1 )) :
            self.networkstructure["layers"][l]["parameters"]['dZ'] = self.networkstructure["layers"][l+1]["parameters"]["W"]*self.deriv_sigmoid(self.networkstructure["layers"][l]['parameters']) @ self.networkstructure["layers"][l+1]["parameters"]["dZ"]
            self.networkstructure["layers"][l]["parameters"]["db"] = self.networkstructure["layers"][l]["parameters"]["dZ"]  
            if l >= 2  :         
                self.networkstructure["layers"][l]["parameters"]["dW"] = self.networkstructure["layers"][l]["parameters"]['dZ'].T * self.networkstructure["layers"][l-1]["parameters"]['a']
            else :
                self.networkstructure["layers"][l]["parameters"]["dW"] = self.networkstructure["layers"][l]["parameters"]['dZ'].T * self.a0
                
    def fowardpropagation(self,X)->None :    
        self.a0 = X
        # Calcuclating the activation for the first layer
        # First Layer : a[0] = X
        self.networkstructure["layers"][1]["parameters"]["Z"] = self.networkstructure["layers"][1]["parameters"]["W"].T @ X + self.networkstructure["layers"][1]["parameters"]["b"]
        self.networkstructure["layers"][1]["parameters"]["a"] = self.sigmoid(self.networkstructure["layers"][1]["parameters"]["Z"])
        # Calculating the activation from the second layer to the last 
        for l in range(2, self.L)  :
            self.networkstructure["layers"][l]["parameters"]["Z"] = self.networkstructure["layers"][l]["parameters"]["W"].T @ self.networkstructure["layers"][l-1]["parameters"]["a"] + self.networkstructure["layers"][l]["parameters"]["b"]
            self.networkstructure["layers"][l]["parameters"]["a"] = self.sigmoid(self.networkstructure["layers"][l]["parameters"]["Z"])
            
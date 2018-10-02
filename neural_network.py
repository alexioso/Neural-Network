class NeuralNetwork:
    def __init__(self, units_per_layer_array, learning_rate):
        """ Args: 
          - units_per_layer_array: an array containing the number of neurons in each layer
                                   its length is equal to the number of layers, including the 
                                   input layer and not including the output layer
                                   
          - learning_rate: eta value for the neural network
        """    
        # initializing member variables
        self.num_hidden_layers = len(units_per_layer_array) - 1
        
        self.weights_array = []
        self.a_array = [None] * self.num_hidden_layers
        self.z_array = [None] * self.num_hidden_layers
        self.eta = learning_rate 
        
        # creating weights for hidden layers
        for u in range(self.num_hidden_layers):                    # number of layers excluding input layer
            num_inputs_curr_layer = units_per_layer_array[u] + 1   # adding bias unit for inputs
            num_outputs_curr_layer = units_per_layer_array[u+1]    # number of units in next layer
            
            # creating matrix of weights for hidden layer(s)
            hidden_weights = np.random.randn(num_inputs_curr_layer, 
                                             num_outputs_curr_layer)
            self.weights_array.append(hidden_weights)
        
        # creating matrix of weights for output
        num_inputs = units_per_layer_array[-1] + 1   # getting number of neurons from last hidden layer and bias
        num_outputs = 1
        
        output_weights = np.random.randn(num_inputs, num_outputs)
        self.weights_array.append(output_weights) # finish initializing weights


    def predict(self, x):
        """ Args:
          - x: an array of length 'units_per_layer_array[0]`
               it is the input value(s) used to make a prediction
               should be a numpy array
        """
        current_layer = np.array([x])
    
        # solving for h-values (hidden layer sums)
        for w in range(self.num_hidden_layers):
            current_layer = add_col_of_ones(current_layer)   # adding ones for bias unit
            
            z = np.dot(current_layer, self.weights_array[w]) # summing weights and "inputs"
            self.z_array[w] = z                              # appending sum of outputs (z)
            
            a = np.maximum(z, 0)                             # applying activation function
            self.a_array[w]                                  # appending activation (a)
            
            current_layer = a
        
        # predicting y_hat without activation function
        current_layer = add_col_of_ones(current_layer) # adding ones for bias unit
        y_hat_weights = self.weights_array[-1]         # getting last weights for y_hat
        y_hat = np.dot(current_layer, y_hat_weights)   # getting 1x1 matrix, which is y_hat
        
        return y_hat[0][0] # returning y_hat value from 1x1 matrix
    
    
    def update(self, x, y):
        """Args:
          - x: an array of length 'units_per_layer_array[0]`
          - y: a float representing the output
        """
        
        ### notation
        # delta = error for the neuron
        # gradient = derivative of loss function wrt weight/bias
    
        ### compute prediction values via forward propogation
        y_hat = self.predict(x) # saves all the a values and z values for each layer
        y_error = y_hat - y     # dL/dy_hat
        
        ### backpropogate error through layers
        delta_array = [None] * (self.num_hidden_layers + 1)
        gradient_array = [None] * self.num_hidden_layers
        
        # calculating neuron errors for layer before y_hat (column vector)
        last_layer_weights = self.weights_array[-1]
        last_delta_layer = y_error * last_layer_weights
        delta_array[self.num_hidden_layers] = last_delta_layer
        
        ## calculating deltas for all layers except last one
        for l in range(self.num_hidden_layers - 1, -1, -1): # backpropogating
            # calculating left-hand side of Hadamard product
            delta_next = delta_array[l + 1]
            weights = self.weights_array[l]
            left_hand_hadamard = np.dot(weights, delta_next)
            
            # calculating right-hand side of Hadamard product
            g_prime = np.sign(z)
            
            # calculating delta with Hadamard product
            delta_curr = np.multiply(left_hand_hadamard, g_prime)
            delta_array[l] = delta_curr
            
        ## calculating gradient for all weights
        for l in range(self.num_hidden_layers): 
            a = self.a_array[l]
            delta = delta_array[l]
            curr_layer_gradient = np.dot(a, delta)
            
            gradient_array[l] = curr_layer_gradient
            
        ## training weights using calculated gradients
        for l in range(self.num_hidden_layers):
            gradient = gradient_array[l]
            weights[l] = weights[l] - (self.eta * gradient)
        
        new_y_hat = self.predict(x)
        return(y - new_y_hat)

from utils import *

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.1):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights_input_hidden = np.random.uniform(-1,1,size=(hidden_nodes, input_nodes))
        self.weights_hidden_output = np.random.uniform(-1,1,size=(output_nodes, hidden_nodes))
        self.bias_hidden = np.ones((hidden_nodes,1))
        self.bias_output = np.ones((output_nodes,1))
        self.learning_rate=learning_rate
        
    def sigmoid(self,x):
        return 1/(1+ math.exp(-x))
    
    def derivate(self,x):
        return x*(1-x)
    
   
    def feedforward(self,input_v):
        sigmoid_vector = np.vectorize(self.sigmoid)
        
        input_vector = input_v.reshape((self.input_nodes,1))
    
        hidden = np.dot(self.weights_input_hidden,input_vector)
        hidden = np.add(hidden, self.bias_hidden)
        hidden = sigmoid_vector(hidden)
    
        output = np.dot(self.weights_hidden_output, hidden)
        output = np.add(output, self.bias_output)
        output = sigmoid_vector(output)
    
        return output
    
    def backpropagation(self, input_v, target_v):
        input_vector = input_v.reshape((self.input_nodes,1))
        target_vector = target_v.reshape((self.output_nodes,1))
        
        sigmoid_vector = np.vectorize(self.sigmoid)
        derivate_vector = np.vectorize(self.derivate)
    
        hidden = np.dot(self.weights_input_hidden,input_vector)
        hidden = np.add(hidden, self.bias_hidden)
        hidden = sigmoid_vector(hidden)
        
        output = np.dot(self.weights_hidden_output, hidden)
        output = np.add(output, self.bias_output)
        output = sigmoid_vector(output)

        

        output_error = np.subtract(target_vector,output)
        error = output_error.sum(0)
        
        gradient = derivate_vector(output)
        gradient = np.multiply(gradient,output_error)
        gradient = np.multiply(gradient, self.learning_rate)
        
        hidden_transpose = np.transpose(hidden)
        weights_ho_deltas = np.dot(gradient, hidden_transpose)
        
        self.weights_hidden_output = np.add(self.weights_hidden_output, weights_ho_deltas)
        self.bias_output = np.add(self.bias_output, gradient)
        
        
        transpose_weights_hidden_output = np.transpose(self.weights_hidden_output)
        hidden_error = np.dot(transpose_weights_hidden_output, output_error)
        
    
        hidden_gradient = derivate_vector(hidden)
        hidden_gradient = np.multiply(hidden_gradient, hidden_error)
        hidden_gradient = np.multiply(hidden_gradient, self.learning_rate)
        
        input_transpose = np.transpose(input_vector)
        weights_ih_deltas = np.dot(hidden_gradient, input_transpose)
        
        self.weights_input_hidden = np.add(self.weights_input_hidden, weights_ih_deltas)
        self.bias_hidden = np.add(self.bias_hidden, hidden_gradient)

        return error
    

    def train(self, train_dataframe, epochs):
        spam = 0
        ham = 0
        iteration = 0
        error_sample = 200
        errors = []

        for i in range(epochs):
            print("Epoch", i)
            for index, row in train_dataframe.iterrows():
                spam+=(row['label_tag'])  
                input_v = row.to_numpy()
                input_v = input_v[1:len(input_v)-1]
                target_v = np.array([row['label_tag']])
                error = self.backpropagation(input_v, target_v)
                
                if iteration%error_sample == 0:
                    errors.append(error)
                    print("Iteration", iteration, "error", error)
                iteration += 1
            print("\n")
            
        ham = (len(train_dataframe)*epochs)-spam
        print(f"Spam:{spam} - Ham:{ham}")
        print("Done")  

        return np.array(errors)
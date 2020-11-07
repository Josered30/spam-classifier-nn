from neural_network import *

def parse(train_set, unique_words):
    train_data_set = np.zeros((len(train_set),len(unique_words)+1))
    iter = 0

    for index, row in train_set.iterrows():
        inputs = Data.get_inputs_count(row['message'],unique_words)
        inputs = np.append(inputs, int(row['label_tag']))
        train_data_set[iter] = inputs
        iter+=1

    indexes = [i for i in range(iter)]
    columns = copy.deepcopy(unique_words)
    columns.append('label_tag')

    train_dataframe = pd.DataFrame(data=train_data_set, index = indexes, columns= columns)
    train_dataframe.to_csv('train_set.csv')


def test(test_set, nn):
    acuaracy = 0
    confusion_matrix = [[0 for i in range(2)] for i in range(2)]

    for index, row in test_set.iterrows():
        #print(row['message'])
        input_v = Data.get_inputs_count(row['message'],unique_words)
        result = round(nn.feedforward(input_v)[0,0])

        if row['label_tag'] == result:
            acuaracy+=1
        confusion_matrix[row['label_tag']][int(result)] +=1
        #print(row['class'], result,'\n')

    acuaracy /= len(test_set)
    presicion = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1])
    recal = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[1][0])

    print(acuaracy)
    print(presicion)
    print(recal)
    print(confusion_matrix)



def train(unique_words, train_set ,nn):
    parse(train_set,unique_words)
    train_dataframe = pd.read_csv('train_set.csv')
    nn = NeuralNetwork(len(unique_words), (len(unique_words)+1)//2, 1, 0.01)
    errors = nn.train(train_dataframe, 10)

def save(nn, name):
    with open(name, 'wb') as output:
        pickle.dump(nn, output)

def predict(message, unique_words, nn):
    input = Data.get_inputs_count(message, unique_words)
    return round(nn.feedforward(input)[0,0])
     

if __name__ == '__main__':
    spam_ham_document_reader = DocumentReader('spamham.csv')
    unique_words = spam_ham_document_reader.get_words()
    unique_words_df = pd.DataFrame(unique_words, columns=['word'])
    unique_words_df.to_csv('words.csv')

    df = pd.read_csv('spamham.csv')
    df["label_tag"] = df["class"].map({'ham':0, 'spam':1})

    train_set = df.sample(frac=0.8)
    test_set = df.drop(train_set.index)
   
    #train(unique_words, train_set, nn)
    #test(test_set, nn)
    #save(nn, 'nn.pkl')

    output = open('nn.pkl','rb')
    nn = pickle.load(output)
    output.close()

    print(predict('How long before you get reply, just defer admission til next semester', unique_words ,nn))




    
    
   
    


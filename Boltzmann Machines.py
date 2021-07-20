import numpy as np
import pandas as pd
import torch as torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import variable

movies = pd.read_csv(r'c:\Users\JAVED\OneDrive\Documents\GitHub\Simple_Titanic_ANN\.vscode\databases\ml-1m\movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv(r'c:\Users\JAVED\OneDrive\Documents\GitHub\Simple_Titanic_ANN\.vscode\databases\ml-1m\users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv(r'c:\Users\JAVED\OneDrive\Documents\GitHub\Simple_Titanic_ANN\.vscode\databases\ml-1m\ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#The data is weird so we had to put some more stuff
training_set = pd.read_csv(r'c:\Users\JAVED\OneDrive\Documents\GitHub\Simple_Titanic_ANN\.vscode\databases\ml-100k\u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv(r'c:\Users\JAVED\OneDrive\Documents\GitHub\Simple_Titanic_ANN\.vscode\databases\ml-100k\u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0]))) #gets the maximum number of users since column 0 is the users, we put those two next to each other since the max can be in either one of them, if so it prints it out
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1]))) #gets the maximum number of movies since column 0 is the users, we put those two next to each other since the max can be in either one of them, if so it prints it out
def convert(data): #we must make it a function because pytorch expects a list of lists, that is what we will do right now
    new_data = []
    for id_users in range(1, nb_users + 1): #we do this range because we want to go over for each users
        id_movies = data[:,1][data[:,0] == id_users] #gets the list of movies rated by the user, gets the first index of the training set and the additional brackets is the condition
        id_ratings = data[:,2][data[:,0] == id_users] #all the ratings by the user
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings #replaces the zeroes with their ratings
        new_data.append(list(ratings)) #this gives us a list of ratings because that is what PyTorch expects
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
#Now, since we will convert to 0 and 1 for positive or negative, the pre existing 0's must now be a different value, -1
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1 
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1 
class RBM():
    def __init__(self, nv, nh): #nv is number of visible nodes, hn is number of hidden nodes, we initialize the weights and the bias, we initialize the parameters that will be optimized in the RBM
        self.W = torch.randn(nh, nv) #randn allows us to initialize with normal distribution of 0 and 1, mean 0 variance 1
        self.a = torch.randn(1, nh) #pytorch can only accept 2 dimension tensors, the hn refers to the bias
        self.b = torch.randn(1, nv) #bias for the visible nodes
    def sample_h(self, x): #must calculate probability of hidden nodes given visible nodes, this returns samples of hidden nodes of our RBM, activates to somme probability, x is the visible neurons
        wx = torch.mm(x, self.W.t())    
        activation = wx + self.a.expand_as(wx) #Each input vector treated as batches since it's a list of a list, the bias must be applied to each line of the mini batch, expand keeps it the same as wx
        p_h_given_v = torch.sigmoid(activation) #the h is the movie genre
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    #Contrastive Divergence: Since RBM is energy based model, we must optimize the weights to minimize the loss, we must get the gradient, approximate because it is so big
    def train(self, v0, vk, ph0, phk): #v0 is the input vector containing the ratings by one user, vk is the visible nodes after k samplings in contrastive divergence, ph0 is the vector of probabilities that at first iteration, probabability is 1 given v0, phk is probability of hidden nodes given visible nodes 
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)) #Updates the weights
        self.b += torch.sum((v0 - vk), 0) #PV Given H, updates that bias. the 0 keeps it as a tensor of two dimensions
        self.a += torch.sum((ph0 - phk), 0) #PH Given V, updates that bias. the 0 keeps it as a tensor of two dimensions
#Visible nodes are the number of movies
nv = len(training_set[0]) #safer route, but nb.movies also works
nh = 1682 #hidden nodes correspond to some features that will be detected, tunable
batch_size = 100 #we update weights after several observations, they go into a batch, tunable
rbm = RBM(nv, nh) #creates the class
#Now we will do the training
number_of_epochs = 10
#Loss: We can use two here, one is Root Mean Square Error (autoencoders), it's the root of the mean of the square differences, or we can use simple (absolute) difference 
for epoch in range(1, number_of_epochs + 1):
    train_loss = 0 #we need a counter to normalize the train_loss
    counter = 0. #float
    for id_user in range(0, nb_users - batch_size, batch_size): #last one is 943 - 100, the step size will be batch_size
        vk = training_set[id_user:id_user+batch_size] #input batch of 100
        v0 = training_set[id_user:id_user+batch_size] #ratings of the movies, we want to compare, but the target is the same as the input at the beginning
        ph0,_ = rbm.sample_h(v0) #we want the first element of what sample_h returns, ph0 is the probability that the hidden node at the start is 1 give the real ratings, v0 is the visible nodes at the start
        for k in range(10): #this is the round trip from visible nodes to hidden nodes
            _,hk = rbm.sample_h(vk) #from sample h we want the second element, hk is hidden nodes at the kth step of contrastive divergence, vk will be updated
            _,vk = rbm.sample_v(hk) #we must update vk, sampled visible nodes after first step of Gibbs Sampling
            vk[v0<0] = v0[v0<0] #we make sure the training is not done on non-existent ratings, make sure they keep the -1 ratings
        phk,_ = rbm.sample_h(vk) #applied on the last sample of the visible nodes
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0])) #loss for the ratings that exist
        counter += 1. #updates the counter
    #print('epoch: ', str(epoch), ' loss: ', str(train_loss / counter)) #normalizes the train_loss

#Batch size training is only for the training set
test_loss = 0 #we need a counter to normalize the train_loss
counter = 0. #float
for id_user in range(nb_users): #last one is 943 - 100, the step size will be batch_size
    v = training_set[id_user:id_user+1] #we need training set to activate hidden neurons to predict the test set
    vt = test_set[id_user:id_user+1] #ratings of the movies, we want to compare, but the target is the same as the input at the beginning
    #ph0,_ = rbm.sample_h(v0) # we don't need this for test set it is only for training
    #for k in range(10): #we only need 1 step for training, since we already did 10, we only need 1
    if len(vt[vt>0]) > 0:
        _,h = rbm.sample_h(v) #from sample h we want the second element, hk is hidden nodes at the kth step of contrastive divergence, vk will be updated
        _,v = rbm.sample_v(h) #we must update vk, sampled visible nodes after first step of Gibbs Sampling
        #vk[v0<0] = v0[v0<0] #we only considered this for training
    #phk,_ = rbm.sample_h(vk) #updates for training
    #rbm.train(v0, vk, ph0, phk) #updates for training
    test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0])) #loss for the ratings that exist
    counter += 1. #updates the counter
print('Test loss: ', str(test_loss / counter)) 






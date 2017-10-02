##################################
# Adrian Soto
# 11-04-2017
# Stony Brook University
#
#
#  
##################################
from __future__ import print_function
from math import floor
import numpy as np
import tensorflow as tf

# Run Agg backend so plots can be generated via ssh without -X
import matplotlib as mpl
mpl.use('Agg')                  
import matplotlib.pyplot as plt
##################################



##################################
# FUNCTIONS
##################################
def int2onehot(num_class, t_in, valid_targets=[0,1]):
    #
    # Convert target value from integer
    # to one-hot.
    #
    #
    # VARIABLES
    # 
    #  INPUT
    #    num_class : number of classes & size of one-hot vector 
    #    t_in      : Nx1 matrix of target values (integers)
    #
    #  OUTPUT
    #    t_out     : Nxnum_class matrix of one-hot vectors 
    #                                                                                                                                                          
    N=len(t_in) # Number of data points 
    
    # Initialize to zero 
    t_out=np.zeros((N, num_class), dtype=int)
    

    # Change value to one where needed
    for i in range(0,N):
        
        # For this data point, find class
        TrueOrFalse = (valid_targets == t_in[i])
                   
        # Convert list of True/False into 1/0 and
        # save into output array of one-hot vectors.
        t_out[i] = 1*TrueOrFalse

    return t_out


def balance_data(x,t):
    # Balance data.
    #   x: np.array containing features
    #   t: np.array containing targets
    
    
    # Check that dimensions agree
    if (np.shape(x)[0] != np.shape(t)[0]):
        print ("ERROR! Lengths of x and t do not agree. Exiting ... ")
        exit()
    
    # Find indices
    iHD=np.array(np.where(t == -1))
    iLD=np.array(np.where(t == +1))
    
    # Count HD and LD
    NHD=np.size(iHD)
    NLD=np.size(iLD)
    
    
    # Create balanced set
    if(NLD <= NHD):
        # Take all LD points and NLD random HD points
        Nhalf=NLD
        ibal=np.zeros(2*Nhalf, dtype=np.int)
        
        ibal[:Nhalf]=iLD[0,:]
        np.random.shuffle(iHD)
        
        ibal[Nhalf:]=iHD[0,:Nhalf]
        
    elif(NLD > NHD):
        # Take all HD points and NHD random LD points
        Nhalf=NHD
        ibal=np.zeros(2*Nhalf, dtype=np.int)
        
        ibal[:Nhalf]=iHD[0,:]
        np.random.shuffle(iLD)
        
        ibal[Nhalf:]=iLD[0,:Nhalf]
        
    
    # Shuffle points so that HD and LD are mixed
    np.random.shuffle(ibal)
    

    # Return shuffled balanced data 
    return x[ibal,:], t[ibal]


#########################
# Read and arrange data
#########################



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%     FEATURE INDICES
#% 1:5   : Voronoi volume, surface, numbers of vertices, edges and sides
#% 6:7   : q, Sk, 
#% 8:20  : LSI0, LsI1, ... LSI12
#% 21:37 : 17 O-O distances
#% 38:42 : 5 O-H-H angles
#% 43:48 : 6 O-O-O angles
#% 49:50 : Nacc, Ndon
#% 51:63 : Number of H-bonds loops of length 3, 4, ..., 15
#iftr          = [1:58] %1:63;   % indices of features to be used
#iftrlabel     ='[1:58]'
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

datadir='./data/'
datafile='D300K.dat'
alldata=np.loadtxt(datadir + datafile)


ifeat=range(1,59)
#ifeat=[1,2,3,4,5,6,7,8,49,50,51,52,53,54,55,56,57,58]
print("Feature indices:", ifeat)
Nfeat=len(ifeat)

x_all=alldata[:,ifeat]
t_all=alldata[:,0]


# Balance data
x_all, t_all = balance_data(x_all, t_all)




# Split dataset into training and test sets
Ndata=np.size(t_all)
Ntes=int(floor(Ndata/5)) # 1/5 of data for test data                                                                                                                              
Ntra=Ndata-Ntes


# Shuffle dataset
ishuff=np.arange(0, Ndata)
np.random.shuffle(ishuff)
itra=ishuff[0:Ntra]
ites=ishuff[Ntra:Ntra+Ntes]


training_set    = x_all[itra, :]
training_labels = t_all[itra]
test_set        = x_all[ites, :]
test_labels     = t_all[ites]

training_labels = int2onehot(2, training_labels.reshape(Ntra), valid_targets=[-1,1])
test_labels     = int2onehot(2, test_labels.reshape(Ntes), valid_targets=[-1,1])


#####################################################
#
#
# Network definition

# Parameters
learning_rate = 0.001
training_epochs = 250
batch_size = 100
display_step = 5


train_num_examples = Ntra

# Network Parameters
n_hidden_1 = 80  # 1st layer number of features
n_hidden_2 = 30  # 2nd layer number of features 
n_input = Nfeat  # Number of features
n_classes = 2    # total classes 

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])



def multilayer_perceptron(x, weights, biases):
    # Hidden layer with sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # Hidden layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer



# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

costs=[]
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train_num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            iL_batch = i*batch_size
            iR_batch = (i+1)*batch_size
            batch_x = training_set[iL_batch:iR_batch, :]
            batch_y = training_labels[iL_batch:iR_batch, :]

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=",                 "{:.9f}".format(avg_cost))
            
            costs.append(avg_cost) # Save costs to evaluate convergence
            
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: test_set, y: test_labels}))
    


    # Errors, classification bias, and other interesting measures
    num_HD_total = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(y, 1), 0), "float"))
    num_LD_total = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(y, 1), 1), "float"))

    NHD = num_HD_total.eval({x: test_set, y: test_labels})
    NLD = num_LD_total.eval({x: test_set, y: test_labels})


    # Count misclassified data points and get classification bias
    predminusy= tf.add(tf.argmax(pred, 1), -1*tf.argmax(y, 1))
    num_HD_misclassified = tf.reduce_sum(tf.cast(tf.equal(predminusy, +1), "float"))
    num_LD_misclassified = tf.reduce_sum(tf.cast(tf.equal(predminusy, -1), "float"))
    
    MHD = num_HD_misclassified.eval({x: test_set, y: test_labels})
    MLD = num_LD_misclassified.eval({x: test_set, y: test_labels})
    
    epsHD = MHD/NHD # HD classification error
    epsLD = MLD/NLD # LD classification error

    cbias = epsLD - epsHD # classification bias
    #print("cbias = ",cbias.eval({x: test_set, y: test_labels}))
    
    print("NHD   = ", int(NHD))
    print("NLD   = ", int(NLD))
    print("MHD   = ", int(MHD))
    print("MLD   = ", int(MLD))
    print("epsHD = ", epsHD)
    print("epsLD = ", epsLD)
    print("cbias = ", epsLD-epsHD)
   
    


epochs= range(1, training_epochs+1, display_step)
plt.figure()
plt.plot(epochs, costs, 'ob')
plt.xlabel('training epoch')
plt.ylabel('cost')
plt.savefig('NN_TrainingCost.pdf')

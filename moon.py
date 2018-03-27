import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #關閉警告訊息

def main():
	m=1000
	x, y = datasets.make_moons(n_samples=m,noise=0.1)
	#add bias for x
	x = np.c_[np.ones((m, 1)), x]  #np.c_是按列连接两个矩阵(就是把两矩阵左右相加)类似于pandas中的merge()
	# y = y.transpose(1,0) #can't use transpose it would become tuple
	y=y.reshape(-1, 1) 
	#Split train data & test data
	test_ratio = 0.2
	test_size = int(m*test_ratio)
	x_train = x[:-test_size]
	x_test = x[-test_size:]
	y_train = y[:-test_size]
	y_test = y[-test_size:]
	data_train = np.c_[x_train,y_train] #concat xtrain & ytrain [1 0.81 0.34 0] data_train[-1] is ytrain
	#but random.shuffle(data_train) would generate 不存在的值 BUGGGG
	y_proba_val=batch_gradient(x_train,y_train,x_test,y_test,m)
	y_real,y_pred,precition,recall= prec_recall(y_test,y_proba_val)
	print(precition,recall)
	plot(x_test,y_real,1)
	plot(x_test,y_pred,2)
	plt.show()
	

def random_batch(x_train,y_train, batch_size):   #(low, high=None, size=None)) low为下限，size为生成的array大小
    rnd_indices = np.random.randint(0, len(x_train), batch_size) #ex [1,4,6,12,32]
    x_batch = x_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return x_batch, y_batch

def batch_gradient(x_train,y_train,x_test,y_test,m):
	tf.reset_default_graph()
	n_inputs = 2
	#當數據量大時，使用constant直接導入數據顯然是不現實的，因而我們用placeholder
	x = tf.placeholder(tf.float32,shape=(None,n_inputs+1),name="x")
	y = tf.placeholder(tf.float32,shape=(None,1),name="y")
	theta=tf.Variable(tf.random_uniform([n_inputs+1,1],-1.0,1.0),name="theta")
	logits = tf.matmul(x,theta,name="logits")
	y_prob = tf.sigmoid(-logits) #sigmoid = 1 / (1 + tf.exp(-logits))
	#log_loss就是算邏輯回去的那一個cost function
	loss = tf.losses.log_loss(y, y_prob)  # uses epsilon = 1e-7 by default to avoid an overflow when computing the log
	learning_rate = 0.02
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss)
	init = tf.global_variables_initializer()
	n_epochs = 1000
	batch_size = 100
	n_batches = int(np.ceil(m / batch_size))

	with tf.Session() as sess:
	    sess.run(init)
	    for epoch in range(n_epochs):
	        for batch_index in range(n_batches):
	            x_batch, y_batch = random_batch(x_train, y_train, batch_size)
	            sess.run(train_op, feed_dict={x: x_batch, y: y_batch})
	        loss_val = loss.eval({x: x_test, y: y_test})
	        if epoch % 100 == 0:
	            print("Epoch:", epoch, "\tLoss:", loss_val)

	    y_proba_val = y_prob.eval(feed_dict={x: x_test, y: y_test})
	return y_proba_val

def prec_recall(y_test,y_proba_val):
	y_test=(y_test>=0.5)
	y_pred=(y_proba_val>=0.5) #if(y>=0.5)->true else:false
	tp=0
	fp=0
	fn=0
	for t, p in zip(y_test, y_pred):
		if t==True and p==True:
			tp+=1
		elif t==False and p==True:
			fp+=1
		elif t==True and p==False:
			fn+=1
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	#from sklearn.metrics import precision_score, recall_score
	#import this then you can get presision in 1 line
	# pscore=precision_score(y_test, y_pred)
	return y_test,y_pred,precision,recall


def plot(x,y,plotnum):
	plt.subplot(1,2,plotnum)
	y=y.reshape(-1)
	plt.plot(x[y , 1], x[y , 2], 'go', label="Positive")
	plt.plot(x[~y , 1], x[~y , 2], 'r^', label="Negative")
	plt.legend() #圖例
	

if __name__ == "__main__":
    main()
#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.utils.data as Data
import ipdb

EPOCH = 20					#训练整批数据多少次，为了节约时间，训练一次
BATCH_SIZE = 50
LR = 0.0001					#学习率



def Dataframe_to_Array_reshape(x):
	data = []
	x = np.array(x)
	for i in range(len(x)):
		data.append(x[i].reshape(28,28))
	return np.array(data)
class CNN(nn.Module):
	def __init__(self):
		super(CNN,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=1,		#input height
				out_channels=20,	#n_filters
				kernel_size=5,		#filter size
				stride=1,			#filter movement/step
				padding = 2,		#if want same width and length of this image after con2d,pading = (kernel_size-1)/2 if stride = 1
			),
			nn.ReLU(),
			# nn.Sigmoid(),
			nn.MaxPool2d(kernel_size=2),

		)
		self.bn1 = nn.BatchNorm1d(num_features=20)
		# self.drop = nn.Dropout(p=0.3)
		self.conv2 = nn.Sequential(
			nn.Conv2d(20,40,5,1,2),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		self.bn2 = nn.BatchNorm1d(num_features=40)
		# self.drop = nn.Dropout(p=0.2)
		self.conv3 = nn.Sequential(
			nn.Conv2d(40,60,5,1,2),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.bn3 = nn.BatchNorm1d(num_features=60*3*3)
		# self.drop = nn.Dropout(p = 0.1)
		self.out1 = nn.Linear(60*3*3,300)	#fully connected layer,output 10 classes
		self.outs = nn.Sigmoid()
		self.out2 = nn.Linear(300, 10)  # fully connected layer,output 10 classes
		self.drop = nn.Dropout(p = 0.4)

	def forward(self,x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = x.view(x.size(0),-1) 	#flatten the output of conv2 to (batch_size,32*7*7)
		output = self.out1(x)
		output = self.outs(output)
		output = self.out2(output)
		return output,x

cnn = CNN()
if torch.cuda.is_available():
	cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()
if __name__ == "__main__":
	data = pd.read_csv("../data/train.csv")
	train_y = data['label']
	# train_x = data.ix[:,1:]
	train_x = data.drop('label', axis=1)

	train_x = Dataframe_to_Array_reshape(train_x)
	train_y = np.array(train_y)

	x_tensor = torch.Tensor.float(torch.from_numpy(train_x))
	y_tensor = torch.Tensor.long(torch.from_numpy(train_y))
	train_data = Data.TensorDataset( x_tensor, y_tensor)

	train_loader = torch.utils.data.DataLoader(dataset = train_data,batch_size = BATCH_SIZE,shuffle = True)

	for epoch in range(EPOCH):
		for step,(x,y) in enumerate(train_loader):
			if torch.cuda.is_available():
				x = x.cuda()
				y = y.cuda()
			output = cnn(x.unsqueeze(1))[0]
			loss = loss_func(output,y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if step % 100 == 0:
				print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])


	data_text = pd.read_csv("../data/test.csv")
	test_x = Dataframe_to_Array_reshape(data_text)
	x_tensor = torch.Tensor.float(torch.from_numpy(test_x) )
	test_loader = torch.utils.data.DataLoader(dataset = x_tensor,batch_size = BATCH_SIZE)

	result = []
	for step,(x) in enumerate(test_loader):
		if torch.cuda.is_available():
			x = x.cuda()
		output = cnn(x.unsqueeze(1))[0]
		output = output.cuda().data.cpu()
		pred_y = torch.max(output,1)[1].data.numpy().squeeze()
		result.append(pred_y.tolist())
	result = np.array(result).reshape(-1)
	id_list = [(i+1)for i in range(len(data_text))]
	pred = np.vstack((np.array(id_list).reshape(-1),result)).T
	print(pred.shape)
	np.savetxt('../data/result_LetNet_Normaliz_BN_Dropout_20_0.0001.csv',pred,fmt=['%s']*pred.shape[1],delimiter=',',newline='\n',header='ImageId,Label')


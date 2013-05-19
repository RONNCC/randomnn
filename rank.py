from pybrain.tools.shortcuts import buildNetwork
import numpy as np
from time import sleep
from skimage.transform import resize
import os
from scipy.misc import imread
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

root = os.getcwd()
has = root+'/has'
noth = root+'/not'
test = root+ '/test'
size=300
ds = SupervisedDataSet(size,1)
net = buildNetwork(size,size/2,size/4,size/8,size/16,size/32,1)
#print net.activate([2,1])

#hot

#makes it compatible as an input to the net
def make_input(f):
    if len(f)%2!=0:
        f=f[:-1]    
    try:
        f=np.reshape(f,(-1,2))
    except ValueError as e:
        print 'reshape error',f.shape,e
        exit(1)
    f = resize(f,(size,1))
    f=np.ndarray.flatten(f)
    return f

for haspath in os.listdir(has):
    f = np.ravel(imread(has+'/'+haspath))
    ds.addSample(make_input(f),(1,))    
for notpath in os.listdir(noth):
    f = np.ravel(imread(noth+'/'+notpath))
    ds.addSample(make_input(f),(0,))    
print 'added pictures'

trainer = BackpropTrainer(net,ds)
print 'testing'
for x in range(5):
    print x,trainer.train()

print 'scoring'
for testpath in os.listdir(test):
    f = np.ravel(imread(test+'/'+testpath))
    print '{}: '.format(testpath),net.activate(make_input(f))

print 'done'


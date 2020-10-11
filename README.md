# tfQuantumJs
An Attempt to combine TensorflowJS with Quantum Javascript



## Tuesday Oct 6th, 2020. This will probably not work, but fun to try.


## Thursday Oct 8th, 2020
I didn't get TensorflowJS working with Tensorflow Quantum, but I did get Tensorflow Quantum working in a gitpod.

but first load the 

## Gitpod of this Github at (right click open in new window, needs you to have a github login) https://gitpod.io/#github.com/hpssjellis/tfQuantumJs

Note: This gitpod installs everything and autoloads the entire googleCollab in one file, which seems to work. The one file is called a01-mnist-small.py


# Installation

The gitpod installs everything and starts the mnist-quantum example.
You can ctlr-C to stop it if you want.
It or other examples when made can be run with
```
python3 a01-mnist-small.py

```
For simplicity line 200 has been commented out for a shorter number of batchs = 3 change it back for the longer run

https://github.com/hpssjellis/tfQuantumJs/blob/53f6629c36e4bc939aa960302a4db463f903ebc3/a05train01.py#L200-L201


Having some issues getting matlablib to make plots while running.

Testing to see if these work: Mnist full should take about 20 min or more.
```
python3 a02-mnist-full.py

```
Testing

```
python3 a03-gradients.py
python3 a04-hello-many-worlds.py
python3 a05-qcnn.py

```
Note: a tf.keras.utils.plot_model command is not working, for the last 2 programs hello and qcnn. I will be looking into it.



To make the above python program I converted the ipython notebook by opening up the 
folders quantum --> docs --> tutorials and typing

```

jupyter nbconvert --to script mnist.ipynb

```
It will output mnist.txt and you can change it to mnist.py. A few of the first commands need to be changed.







## This Repo website https://hpssjellis.github.io/tfQuantumJs/public/index.html  


Batch of tweets: 


https://twitter.com/rocksetta/status/1312604679017099271?s=19


Mnist Google collab

https://www.tensorflow.org/quantum/tutorials/mnist

https://colab.research.google.com/github/tensorflow/quantum/blob/master/docs/tutorials/mnist.ipynb


IBM actual quantum computer 

https://www.ibm.com/quantum-computing

Symbols Explained
https://quantum-computing.ibm.com/docs/iqx/operations-glossary


Simulators:

https://quantumjavascript.app/












First Issue. Q.js is a playground, can't seem to figure out how to make my own javascript pages






https://raw.githubusercontent.com/stewdio/q.js/master/build/q.js



https://github.com/stewdio/q.js



Click on the image to be able to zoom  
[![IBM-Symbols](ibm-symbols-and-names05.png)](https://hpssjellis.github.io/tfQuantumJs/public/ibm-quantum-symbols.html)





Something interesting at  https://github.com/google/jax but I have not yet found out if JAX will be useful for me.

```
pip3 install jax

git clone https://github.com/google/jax.git

```


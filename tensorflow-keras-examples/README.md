To get going with quantum I need to remind myself about regular tensorflow (I am used to tensorflowJS). Here are some examples.


Once again I am just taking the regular examples

These ones are from https://github.com/tensorflow/docs/tree/master/site/en/tutorials/keras using

```
git clone https://github.com/tensorflow/docs/tree/master/site/en/tutorials/keras

```
Then converted to python using

```
jupyter nbconvert --to script fileName.ipynb

```

Adding a few changes like

```
# visualization tools
#%matplotlib inline
import matplotlib
matplotlib.use('TkAgg')

```

and 
```
#plt.show()
plt.draw()
plt.pause(0.001)
input("Open Ports --> Open Preview or Browser --> push enter to continue")

```


and maybe commenting out some tf charting.


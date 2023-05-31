# HyperKnee Finder

This tool will found the optimal values for two inter-dependant parameters using the well known knee/elbow method.

The method prescribe to search for the point where the curvature is at the maximum ([Here an example](https://en.wikipedia.org/wiki/Elbow_method_(clustering))).

A more formal definition (from Satopää, Albrecht, Irwin, and Raghavan, 2011, p.1) states that the knee/elbow point is the point after which:

> relative costs to increase [or decrease, NdC] some tunable parameter is no longer worth the corresponding performance benefit

While usually this method is used for tuning one single parameter, nothing impeach to the same for  multiple, 
inter-dependant parameters. This tool is anyway able to find the optimal value only for two parameters.

![Plot of the HyperKnee](https://github.com/vlavorini/hyperknee_finder/blob/main/notebooks/hk_plot.png?raw=True)

In facts, HyperKnee Finder is a 2-d generalisation of the [KneeFinder](https://github.com/vlavorini/kneefinder) tool.

## Motivations for HyperKnee Finder
In many situations the parameters of an algorithm depends on each other.  What you usually do is to ignore this 
dependency, and so you optimise the first parameter, then you use the found value to optimise the second parameter. 

Acting like this, in general, does not guarantee you to land on the optimal combination for the two parameters.

Indeed, you have to evaluate all the possible combination of the two parameters, and at that point
you can make your choice.


## Examples
The following example show how to find the HyperKnee point in a double exponential decay.
```python
from hyperkneefinder import HyperKneeFinder
import numpy as np
%matplotlib ipympl

#double exponential decay plus noise, clipped
X = np.arange(1, 8, 0.1)

Y = np.arange(6, 10, 0.1)
Z = np.zeros((len(X), len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        Z[i, j] = np.exp(-X[i]/3) + np.exp(-(Y[j]-5)) + np.random.rand()/45

Z = np.clip(Z, a_min= 0.5, a_max=2)
hkf= HyperKneeFinder(X, Y, Z, name_x='parameter_1', name_y='parameter_2', clean_data=True, clean_threshold=0.8)
hkf.visualise_hyperknee()
```

![Plot of the HyperKnee](https://github.com/vlavorini/hyperknee_finder/blob/main/notebooks/hk_plot2.png?raw=True)

Also with different pseudo-convexity:

```python

X = np.arange(1, 8, 0.1)

Y = np.arange(6, 10, 0.1)
Z = np.zeros((len(X), len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        Z[i, j] = np.exp(-2/X[i]) + np.exp(-3/(Y[j]-5)) # + np.random.rand()/15
        
Z = np.clip(Z, a_min= 0, a_max=1)
hkf= HyperKneeFinder(X, Y, Z, name_x='parameter_1', name_y='parameter_2', clean_data=True, clean_threshold=0.8)
hkf.visualise_hyperknee()
```


![Plot of the HyperKnee](https://github.com/vlavorini/hyperknee_finder/blob/main/notebooks/hk_plot3.png?raw=True)
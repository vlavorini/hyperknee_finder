# HyperKnee Finder

This tool will found the optimal values for two inter-dependant parameters using the well known knee/elbow method.

The method prescribe to search for the point where the curvature is at the maximum ([Here an example](https://en.wikipedia.org/wiki/Elbow_method_(clustering))).

A more formal definition (from Satopää, Albrecht, Irwin, and Raghavan, 2011, p.1) states that the knee/elbow point is the point after which:

> relative costs to increase [or decrease, NdC] some tunable parameter is no longer worth the corresponding performance benefit

While usually this method is used for tuning one single parameter, nothing impeach to the same for  multiple, inter-dependant parameters.

## Motivations for this tool
In many situations the parameters of an algorithm depends on each other. This means that tuning each parameter 
independently, and even optimising parameters in cascade, results in a non-optimal result.

Say you have THIS data, and you want to build clusters around them with DBScan. You want to optimise the two most 
important parameters: Eta and Min_samples.

What you usually do is to optimise the first parameter, then you use the found value to optimise the second parameter. 
Let's see what happens in an example.


![Plot of the HyperKnee](./notebooks/hk_plot.png)
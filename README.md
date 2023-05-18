# HyperKnee Finder

This tool will found the optimal values for two inter dependant parameters using the well known knee/elbow method

The method prescribe to find the point where the curvature is at the maximum ([Here an example](https://en.wikipedia.org/wiki/Elbow_method_(clustering))).

A more formal definition (from from Satopää, Albrecht, Irwin, and Raghavan, 2011, p.1) states that the knee/elbow point is the point after which:

> relative costs to increase [or decrease, NdC] some tunable parameter is no longer worth the corresponding performance benefit

While usually this method is used for tuning one single parameter, nothing impeach to the same for  multiple, inter-dependant parameters.

## Motivations for this tool
In such situations, indeed, in almost all the cases the parameters of the same algorithm depends on each other, i.e. ... .
This means that tuning each parameter independently of the others gives a wrong result.


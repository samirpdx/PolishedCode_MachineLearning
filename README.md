PolishedCode_MachineLearning
=============================

Coordinate Descent Algorithm with Elastic Net Regularization
--------------------------------------------------------------

This is a polished code example of my own implementation of a Coordinate Descent Algorithm with Elastic Net 
Regularization used for solving least-squares regression for the minimization problem seen below:  

<img src=https://github.com/samirpdx/PolishedCode_MachineLearning/blob/master/images/elasticnet.JPG width="600" height="80" />

This package was created as a part of my DATA 558 Machine Learning course at the University of Washington.

For examples of implementation, please see 
the [examples](https://github.com/samirpdx/PolishedCode_MachineLearning/tree/master/examples) folder.  

No data files are required for download, as they are downloaded into the notebook via URLs.

For viewing the raw Python code for this implementation, please refer 
to [myelasticnet.py](https://github.com/samirpdx/PolishedCode_MachineLearning/blob/master/src/myelasticnet.py) 

Directory Structure
---------------------
```
PolishedCode_MachineLearning/

  |- examples/
     |- __init__.py
     |- README.md
     |- Polished Code - ElasticNet (Comparison with Sci-Kit Learn).ipynb
     |- Polished Code - ElasticNet (Real-World Example).ipynb
     |- Polished Code - ElasticNet (Simulated Example).ipynb
  |- images/
     |- elasticnet.jpg
  |- src/
     |- __init__.py
     |- myelasticnet.py
  |- README.md
  |- setup.py
```

Installation
---------------

_Note:  To run this package you will need familiarity with bash command line and Jupyter Notebook._

In a directory on your local machine, run the following `git` command in the bash terminal to clone the 
`PolishedCode_MachineLearning` repository onto your computer:

```
git clone https://github.com/samirpdx/PolishedCode_MachineLearning.git
```

Then in the bash terminal go to the the newly installed package folder:

```
cd PolishedCode_MachineLearning/
```

And install the package by running the `setup.py` file:

```
python setup.py install
```


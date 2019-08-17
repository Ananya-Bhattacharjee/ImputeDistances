# Description

We introduce two highly accurate machine learning based distance imputation techniques. One of our approaches is based on matrix factorization (MatrixFactorization.py), and the other one is an autoencoder based deep learning technique (Autoencoder.py). We evaluate these two techniques on a collection of simulated and biological datasets, and show that our techniques are more accurate and robust than the best alternate technique for distance imputation. Moreover, our proposed techniques can handle substantial amount of missing data, to the extent where the best alternate method fails.

# Requirement

(See the imports in both of the python files)

-python 3.5+
-easygui
-pandas
-numpy 
-matplotlib
-seaborn
-tensorflow
-keras

# How to Run

~python MatrixFactorization.py

~python Autoencoder.py


Select a distance matrix with missing entries and the complete matrix will be available as the output. An example distance matrix with missing values (Missing.dis) is provided. All the inputs should follow this format. The file should start with the number of taxa and then follow with the distance matrix. A missing entry should be represented with  a dot ('.').  

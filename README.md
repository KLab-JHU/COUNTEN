# COUNTEN
COUNTEN - COUNTing Enteric Neurons
#
1- Download Anaconda, a free installer that includes Python and all the common scientific packages.

2- Clone the repository:

git clone https://github.com/KLab-JHU/COUNTEN
3- Go into the directory:

cd COUNTEN-master/
4- Create a conda environment with an Ipython kernel:

 conda create --name name_env python=3 ipykernel
5- Activate the conda environment:

source activate name_env
6- Install the python modules that will be required for the program:

pip install numpy
pip install javabridge
pip install scikit-image
pip install scikit-learn
pip install pandas
pip install ipykernel
pip install bioformats
pip install python-bioformats
7- Make sure that you are pointing to the correct java folder:

export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-14.0.2.jdk/Contents/Home
Usage
1- Launch jupyter notebook:

jupyter notebook
2- Open LowRes_Analysis.ipynb file and specify the correct location of the image file, pixel density and sigma (smoothening factor)

3- Run the program

Contact
Subhash Kulkarni
E-mail: skulkar9@jh.edu

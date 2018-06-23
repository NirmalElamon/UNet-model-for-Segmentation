
#----------------Run this code for testing the model--------------------#
import os, sys
import configparser as ConfigParser

#Reading the config file which has all the parameters
config = ConfigParser.RawConfigParser()
config.readfp(open(r'./configuration.txt'))

#===========================================
#extracting the name of the experiment
name_experiment = config.get('experiment name', 'name')

run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

#create a folder for the results if not existing already
result_dir = name_experiment
print ("\n1. Create directory for the results (if not already existing)")
if os.path.exists(result_dir):
    pass
elif sys.platform=='win32':
    os.system('md ' + result_dir)
else:
    os.system('mkdir -p ' + result_dir)


# finally run the prediction


print ("\n2. Run the prediction on GPU")
os.system(run_GPU +' python ./src/retinaNN_predict.py')

import sys
# Check that python 3.6 installed
if sys.version_info[0] < 3 and sys.version_info[1] < 6:
    raise Exception("Must be using at least Python 3.6")

try:
    import sklearn
except:
    raise Exception("Requires sklearn package: pip3 install scikit-learn")

try:
    import numpy
except:
    raise Exception("Requires numpy package: pip3 install numpy")

try:
    import scipy
except:
    raise Exception("Requires scipy package: pip3 install scipy")

try:
    import pickle
except:
    raise Exception("Requires pickle package: pip3 install pickle-mixin")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# set warnings setting to ignore mode
import warnings
warnings.filterwarnings("ignore")

#set print format to float format for all numeric outputs to prevent like 'e+06' outputs
pd.set_option('display.float_format', lambda x: '%.3f' % x)
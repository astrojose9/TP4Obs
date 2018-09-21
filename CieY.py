#!/usr/bin/env python
# -*-coding:Utf-8-*

"""
@author: José Rodrigues
"""

#########################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import astropy 
from astropy.table import Table
import scipy
from scipy import constants as const
from scipy.interpolate import interp1d

#########################################################################################################
ciey=Table.read('CIE-Y.csv')
xciev=ciey.columns[0]*1e-9
yciev=ciey.columns[1]
cieY = interp1d(xciev, yciev, kind='cubic')
xciev = np.arange(min(ciey.columns[0]), max(ciey.columns[0]), 0.1)*1e-9
#print(str(min(xciev))+str(max(xciev)))
#plt.plot(xciev,cieY(xciev))
#plt.show()
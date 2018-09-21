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
from scipy.integrate import quad
from scipy.interpolate import interp1d 
from CieY import cieY

#########################################################################################################
Km=683.002 #lm/W

#
def specFill(t,xMin,xMax,step):
	"Adds rows of zeros to fill a spectral range from xMin to xMax. "
	lowSpec=int(min(t.columns[0]))
	upSpec=int(max(t.columns[0]))

	if min(t.columns[0])<xMin: #Checks if xMin is bigger than the min of the spectra and if so defines a lower xMin
		xMin = lowSpec-step
	if max(t.columns[0])>xMax: #Checks if xMax is smaller than the max of the spectra and if so defines a bigger xMax
		xMax = upSpec+step
	xBinsLow=(lowSpec-xMin)/step # number of bins under the lowest value given by the spectra
	for i in np.linspace(xMin,lowSpec,xBinsLow):
		t.add_row(np.zeros(len(t.colnames))) #adds a row full of zeros
		t.columns[0][-1]=i #replaces the 0th element with a wavelength
	xBinsUp=(xMax-upSpec)/step # number of over under the lowest value given by the spectra
	for i in np.linspace(upSpec,xMax,xBinsUp):
		t.add_row(np.zeros(len(t.colnames))) #adds a row full of zeros
		t.columns[0][-1]=i #replaces the 0th element with a wavelength

	t.sort(0) #sorts the table with respect to the wavelegth colunm
	return t

def specClean(t,treshold):
	"brings values under a treshold to zero"
	for key in t.colnames:
		for l in range(len(t.columns[key])):
			if t[key][l]<treshold:
				t[key][l]=0
	return t

#Watt
def flx(l,y):
	"flx(l,y) is basically a proxy for interpol1d"
	flux=interp1d(l,y,kind=3) #In W/m |function
	return flux

def wattNorm(l,y,mW):
	"wattNorm(l,y,mW) takes wavelenght, spectral flux and the mW output given by the manufacturer as parameters and returns a spectral flux in W/m"
	flux = flx(l,y) # |function
	I=quad(flux,min(l),max(l)) #in W
	mWFloat = float(mW)
	C=mWFloat*1e-3/(I[0]) #We renormalise using the previously calculated integral. "mW" is the flux given by the manufacturer in mW..
	return C*y #*1e-6 #In W/m *1e-6 can be used to return it in W/µm

#Lumen
def flxLm(l,y):
	"flxLm(a)=Km*flx(l,y)*cieY(l). It is used to convolute a spectrum with the CieY curve and multiply is with Km=683,002 lm/W so the curve can be integrated to give an output in lumen. l shoud be the wavelenght in m"
	flux=flx(l,y) # |function
	fluxLmarr=Km*flux(l)*cieY(l) #In lm/m |array
	fluxLm=interp1d(l,fluxLmarr,kind=3) #In lm/m |function
	return fluxLm

def lumToWatt(l,y,lm):
	"lumToWatt(l,y,lm) converts a spectral flux from lumen/m to Watt/m. It takes wavelenght, spectral flux and the lumen output given by the manufacturer as parameters and returns a spectral flux in W/m"
	fluxLm=flxLm(l,y) # |function
	I=quad(fluxLm,min(l),max(l)) #In lm. We integrate the spectra convoluted with CieY.
	lmFloat = float(lm)
	C=lmFloat/(I[0]) #We renormalise using the previously calculated integral. "lm" is the flux given by the manufacturer in lm.
	return C*y #*1e-6 #In W/m *1e-6 can be used to return it in W/µm

#Radiance
def fluxToRadiance(y,ledSize, alpha):
	"Takes in a spectral flux in W/m, ledSize in m² and an angle in °"
	sr=2*(1-np.cos(alpha/2))*const.pi #Solid angle obtained from an angle alpha
	return y/(ledSize*sr) #spectral flux is converted to spectral radiance in W m⁻³ sr⁻¹ can be converted in W m⁻² sr⁻¹ µm⁻¹ by multiplying by 1e-6 

def radSmoothing(x,y,newX,treshold):
	"Smooths the radiance by using interp1d, and kills the \"wavy\" artifacts under a certain treshold"
	smoothed=interp1d(x,y,kind=3) 
	newY=smoothed(newX)
	for l in range(len(newY)):
		if newY[l] < treshold:
			newY[l]=0
	return newY
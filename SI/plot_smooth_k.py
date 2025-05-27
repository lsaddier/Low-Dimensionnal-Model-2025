# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:19:37 2024

@author: louis
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib import colors

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def ss(k, d_s, kdiff, kconv, sc):
    dk = kconv-kdiff
    k_mean = (kconv+kdiff)/2
    return sc+d_s*np.arctanh(2*(k-k_mean)/dk)

def kk(s, d_s, kdiff, kconv, sc):
    return (kdiff+kconv)/2+(kconv-kdiff)/2*np.tanh((s-sc)/d_s)

def F_sig(kd, kc, d_s, sc, phi, chi, sig, alpha, x0, g, beta, delta):
    k = kk(sig, d_s, kd, kc, sc)
    numerator = (
        (k * phi + chi)
        * (
            sig
            - (
                alpha * chi * (x0 * chi + g * (-1 + x0) * phi * (1 + chi))
            ) / (
                g * phi * (1 + chi) * (k * phi + chi) + chi * (chi * (1 + chi) + k * phi * (2 + chi))
            )
            + (
                beta * delta * chi * (x0 * chi * (k * phi + chi) + g * phi * (k * phi + x0 * chi))
            ) / (
                (k * phi + chi) * (g * phi * (1 + chi) * (k * phi + chi) + chi * (chi * (1 + chi) + k * phi * (2 + chi)))
            )
        )
    )
    denominator = beta * phi
    return numerator / denominator


def Matrix(kdiff,kconv,chi,phi,g,delta,s,d_s, sc):
    k = kk(s,d_s,kdiff,kconv, sc)
    return np.array([[-chi-phi*k-phi*g, phi*k, chi, 0, 0, 0], 
                     [phi*k, -chi-phi*k, 0, 0, 0, 0],
                     [0,chi, -chi-(1-phi), 0, 0, 0],
                     [0, 0, 0, -chi-phi*k, phi*k, chi],
                     [0, 0, 0, phi*k, -chi-phi*k, 0],
                     [0, 0, -(1-phi)*delta, 0, chi, -chi]])

def Forcing(fs,x0,chi,phi,g):
    return np.array([phi*g, chi*x0, 0, phi*fs, 0 ,0])


def steady(M,F):
    return -np.dot(np.linalg.inv(M),F)


#%%
delta = 5e-3
x0 = 7
chi = 0.5
g = 10
phi = 0.1

kd = 0.005
kc = 50
d_s = 2e-4
sc = 2.5e-4

alpha = 3.45e-5
beta = 2.6e-2

sigmin = -4.5e-4
sigmax = 2e-4
N = 500
list_sig = np.linspace(sigmin,sigmax,N)


fsmin = 0.0
fsmax = 0.1
n = 6
list_fs = np.linspace(fsmin,fsmax,n)

colorlist = [cm.winter(x) for x in np.linspace(0,1,n)]

fig, axs = plt.subplots(1,2, dpi=150, figsize=(7,3))

axs[0].plot(list_sig,list_sig,c='black',linestyle='dashed',label=r'$\sigma=\sigma$')

for i, fs in enumerate(list_fs):
    list_F = []
    for s in list_sig:
        X = steady(Matrix(kd,kc,chi,phi,g,delta,s,d_s, sc),Forcing(fs, x0, chi, phi, g))
        list_F.append(-alpha*(X[0]-X[1])+beta*(X[3]-X[4]))
    axs[0].plot(list_sig,list_F, c=colorlist[i])
    
axs[0].set_xlim((sigmin,sigmax))
axs[0].set_ylim((sigmin,sigmax))

axs[0].set_xlabel(r'$\sigma$')
axs[0].set_ylabel(r'$\mathcal{F}(\sigma)$')
axs[0].legend()

axs[0].ticklabel_format(axis='both',style='sci',scilimits=(-4,-4))

cax = plt.axes((-0.01, 0.14, 0.025, 0.8))
bar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=fsmax,),cmap='winter'),cax=cax, location="left")
bar.ax.set_ylabel(r'$f_S$',size=18)



sigmin = -4.5e-4
sigmax = 2e-4
N = 500


list_sig = np.linspace(sigmin,sigmax,N)
list_fs = F_sig(kd, kc, d_s, sc, phi, chi, list_sig, alpha, x0, g, beta, delta)


sig_thresh = -5e-5
mask0 = np.gradient(list_fs)>0 #apply this mask before mask1 and mask3
mask1 = list_sig[np.gradient(list_fs)>0]<sig_thresh
mask3 = list_sig[np.gradient(list_fs)>0]>sig_thresh
mask2 = np.gradient(list_fs)<0

axs[1].plot(list_fs[mask0][mask1],list_sig[mask0][mask1],c='black',label='Stable')
axs[1].plot(list_fs[mask2],list_sig[mask2],c='black',linestyle='dashed',label='Unstable')
axs[1].plot(list_fs[mask0][mask3],list_sig[mask0][mask3],c='black')
axs[1].ticklabel_format(axis='y',style='sci',scilimits=(-4,-4))

axs[1].set_xlim((fsmin,fsmax))

axs[1].set_ylabel(r'$\sigma$')
axs[1].set_xlabel(r'$f_S$')
axs[1].legend()

plt.tight_layout()

plt.savefig("Figures/smooth_k.pdf", bbox_inches='tight')

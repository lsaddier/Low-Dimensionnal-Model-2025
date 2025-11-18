#################################################################################################
#This is the solver for the buyoancy driven q and constant gamma_T^\star cavity parameterization#
#################################################################################################

import numpy as np

# Define reference properties and parameters 

#EOS
T_star = 0 # °C  (PICO)
S_star = 34 # PSU   (PICO)
rho_star = 1033 # kg/m^3   (PICO)
alpha = 7.5e-5 # /°C   (PICO)
beta = 7.7e-4 # /PSU (PICO)

def EOS(T,S):
    return rho_star*(1-alpha*(T-T_star)+beta*(S-S_star))
    
#Liquidus
la = -0.0572 # °C/PSU   (PICO)
lb = 0.0788 # °C   (PICO)
lc = 7.59e-4 # °C/m   (PICO->Burgard)

def liquidus(S,h):
    return la*S+lb-lc*h

#Water properties
L = 3.34e5 # J/kg   (PICO)
c_star = 3974 # J/kg/°C   (PICO)
lambd = L/c_star # 84 K

#Ice properties
rho_i = 910 # kg/m^3   (PICO)
nu = rho_i/rho_star #0.88


# PICO solver
def PICO(Td,Sd, nbox, C, gammaT, Ac, frac, depth):
    g1 = Ac*frac*gammaT # list of g1
    g2 = g1/nu/lambd
    s = Sd/nu/lambd
    # management of the 1st box
    T_st_0 = la*Sd+lb-lc*depth[0]-Td
    x0 = -g1[0]/(2*C*rho_star*(beta*s-alpha))+np.sqrt((g1[0]/(2*C*rho_star*(beta*s-alpha)))**2-g1[0]*T_st_0/(C*rho_star*(beta*s-alpha)))
    y0 = Sd*x0/nu/lambd
    T0 = Td-x0
    S0 = Sd-y0
    q = C*rho_star*(beta*s-alpha)*x0

    # management of the other boxes
    T = np.zeros(nbox)
    S = np.zeros(nbox)
    m = np.zeros(nbox)
    T[0] = T0
    S[0] = S0
    m[0] = -gammaT/nu/lambd*(la*S[0]+lb-lc*depth[0]-T[0])*3600*24*30 #m/30d
    for k in range(1,nbox):
        T_st = la*S[k-1]+lb-lc*depth[k]-T[k-1]
        x = -g1[k]*T_st/(q+g1[k]-g2[k]*la*S[k-1])
        y = S[k-1]*x/nu/lambd
        T[k] = T[k-1]-x
        S[k] = S[k-1]-y
        m[k] = -gammaT/nu/lambd*(la*S[k]+lb-lc*depth[k]-T[k])*3600*24*30 #m/30d
 
    return T,S,m,q


# compute AABW flux [m^3/s]
def compute_DSW(T, S, C2, T0, S0):
    #return C2*(EOS(T,S)-EOS(T0,S0)) #Ensure non-negative flux
    return np.max(np.array([0,C2*(EOS(T,S)-EOS(T0,S0))])) #Ensure non-negative flux


# compute water column stability [kg/m^3] # work only in diffusive mode
def compute_Sigma(X):
    return EOS(X[0],X[2])-EOS(X[1],X[3]) #rho_p - rho_d

def compute_rhod_rho0(T,S,T0,S0):
    return EOS(T,S)-EOS(T0,S0)

# Compute melt rate averaged over all the cavity boxes [m/30d]
def compute_m_avg(m,frac,nbox):
    return np.average(m, weights=frac[:nbox])



def BoxModel(X, nbox, C, gammaT, Ac, Omega, T0, S0, Ap, kappa, g, frac, depth, r):
    
    #compute k. 
    k = kappa/gammaT

    #compute phi
    phi = Ap/Ac

    #define variables. Here no Tp because Tp=Td at steady state
    Tp, Td, Sp, Sd = X

    # Cavity dynamics with PICO 
    Tc,Sc,m,q = PICO(Td,Sd, nbox, C, gammaT, Ac, frac, depth)


    chi = q/Ac/gammaT

    # create an empty vector that will contains the dynamical system
    vect_out = np.zeros(len(X))

    # Polynya box equations
    vect_out[0] = chi*(Tc[-1]-Tp) + phi*g*(liquidus(Sp,0)-Tp) + phi*k*(Td-Tp)
    vect_out[2] = chi*(Sc[-1]-Sp) + phi*rho_i/rho_star*Sp/gammaT*Omega + phi*k*(Sd-Sp)

    # Deep box equations
    vect_out[1] = r*chi*(T0-Td) + (phi*k+chi*(1-r))*(Tp-Td)
    vect_out[3] = r*chi*(S0-Sd) + phi*k*(Sp-Sd)

    
    return vect_out



# Continuous parameterization of the vertical mixing
def BoxModel_continuous_kappa(X, nbox, C, gammaT, Ac, Omega, T0, S0, Ap, kappa_d, kappa_c, Lamb, g, frac, depth, C2, T_surf, S_surf, r, mode):

    phi = Ap/Ac

    #compute k
    Sigma = compute_Sigma(X)
    kappa = (kappa_c-kappa_d)/2*np.tanh(Sigma/Lamb)+(kappa_c+kappa_d)/2
    k = kappa/gammaT

    #define variables
    Tp, Td, Sp, Sd = X

    Sigma_d_0 = compute_rhod_rho0(Td,Sd,T0,S0)

    # Cavity dynamics with PICO 
    Tc,Sc,m,q = PICO(Td,Sd, nbox, C, gammaT, Ac, frac, depth)

    chi = q/Ac/gammaT
    
    # create an empty vector that will contains the dynamical system
    vect_out = np.zeros(len(X))

    if mode == 'CDW':

        # polynya box equations
        vect_out[0] = chi*(Tc[-1]-Tp) - phi*k*(Tp-Td) + phi*g*(liquidus(Sp,0)-Tp)
        vect_out[2] = chi*(Sc[-1]-Sp) - phi*k*(Sp-Sd) + phi*rho_i/rho_star*Sp/gammaT*Omega
        
        # deep box equations
        vect_out[1] = chi*(T0-Td) + phi*k*(Tp-Td)  + r*(T0-Td)
        vect_out[3] = chi*(S0-Sd) + phi*k*(Sp-Sd) + r*(S0-Sd)
    
    if mode == 'DSW':

        DSW = compute_DSW(Td, Sd, C2, T0, S0)/Ac/gammaT

        # Polynya box equations
        vect_out[0] = chi*(Tc[-1]-Tp) + phi*g*(liquidus(Sp,0)-Tp) + DSW*(T_surf-Tp) + phi*k*(Td-Tp)
        vect_out[2] = chi*(Sc[-1]-Sp) + phi*rho_i/rho_star*Sp/gammaT*Omega + DSW*(S_surf-Sp) + phi*k*(Sd-Sp)

        # Deep box equations
        vect_out[1] = chi*(Tp-Td) + DSW*(Tp-Td) + phi*k*(Tp-Td) + r*(T0-Td)
        vect_out[3] = chi*(Sp-Sd) + DSW*(Sp-Sd) + phi*k*(Sp-Sd) + r*(S0-Sd)




    return vect_out
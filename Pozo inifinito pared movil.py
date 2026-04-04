# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:25:23 2026

@author: July
"""
import numpy as np
from scipy.integrate import solve_ivp as ivp
from numba import njit,complex128
import time 
import matplotlib.pyplot as plt
import shutil as sh

h_bar = 1           #Unidades geometrizadas
infty = 5000        #Esto sirve como infinito


# =============================================================================
# # Parámetros
# =============================================================================

z0 = 0
zf = 1

omega = 1

t0 = 0
tf = 1


m = 100              #Partes espaciales
Nt = 30             #Partes temporales


nombre1 = 'Datos L.dat'
nombre2 = 'Datos S.dat'

caso = int(input('¿Qué caso se quiere resolver? \n 0 : Lineal || 1 : Sinusoidal \n'))



# =============================================================================
# # Funciones
# =============================================================================

# def dxx(x):
#     return

# def V(x,t):
#     global L, infty
    
#     if x > 0 and x < L:
#         return 0
#     else: 
#         return infty

def f(z):
    return 2**0.5*np.sin(2*np.pi*z)
@njit
def Ll(a,b,t): 
    return a + b*t
@njit
def dLl(b):
    return b
@njit
def Ls(a,b,t,omega):
    return a + b*np.sin(omega*t)
@njit
def dLs(b,t,omega):
    return omega*b*np.cos(omega*t)

@njit
# @jit( (complex128[:,:,:],), nopython=True  )
def resl(t,phi):
    global h
    r = len(phi)
    # print(r)
    dphi = np.zeros_like(phi, dtype = complex128)       #recordar que para phi0 y phi-1 son ceros
    
    for i in range(0,r-1):
        if i != 0 and i != r-1:
            # print(i,len(dphi))
            # print('phi_i= ', phi[i], 'phi_{i-1}= ', phi[i-1], 'phi_{i+1}= ',phi[i+1],'||',dphi[i])
            dphi[i] = 1/h**2*1j/Ll(a,b,t)**2*(phi[i+1]-2*phi[i]+phi[i-1]) + 1/2*(i+1)*dLl(b)/Ll(a,b,t)*(phi[i+1] - phi[i-1])
            # dphi[i] = 1/h**2*1j/Ls(a,b,t,omega)**2*(phi[i+1]-2*phi[i]+phi[i-1]) + 1/2/h*i*dLs(b,t,omega)/Ls(a,b,t,omega)*(phi[i+1] - phi[i-1])
        else: 
            if i == 0: 
                # print(i,phi[i])

                dphi[i] = 1/h**2*1j/Ll(a,b,t)**2*(phi[i+1]-2*phi[i]+ 0) + 1/2*(i+1)*dLl(b)/Ll(a,b,t)*(phi[i+1] - 0)
                
            if  i == r-1:
                # print(i,phi[i])
                
                dphi[i] = 1/h**2*1j/Ll(a,b,t)**2*( 0 -2*phi[i]+phi[i-1]) + 1/2*(i+1)*dLl(b)/Ll(a,b,t)*( 0 - phi[i-1])
                
        
    # print(dphi)
    return dphi

@njit
def resc(t,phi):
    global h
    r = len(phi)
    # print(r)
    dphi = np.zeros_like(phi, dtype = complex128)       #recordar que para phi0 y phi-1 son ceros
    
    for i in range(0,r-1):
        if i != 0 and i != r-1:
            # print(i,len(dphi))
            # print('phi_i= ', phi[i], 'phi_{i-1}= ', phi[i-1], 'phi_{i+1}= ',phi[i+1],'||',dphi[i])
            # dphi[i] = 1/h**2*1j/Ll(a,b,t)**2*(phi[i+1]-2*phi[i]+phi[i-1]) + 1/2/h*i*dLl(b)/Ll(a,b,t)*(phi[i+1] - phi[i-1])
            dphi[i] = 1/h**2*1j/Ls(a,b,t,omega)**2*(phi[i+1]-2*phi[i]+phi[i-1]) + 1/2*(i+1)*dLs(b,t,omega)/Ls(a,b,t,omega)*(phi[i+1] - phi[i-1])
        else: 
            if i == 0: 
                # print(i,phi[i])
                # print('Calculando...')

                # dphi[i] = 1/h**2*1j/Ll(a,b,t)**2*(phi[i+1]-2*phi[i]+ 0) + 1/2/h*i*dLl(b)/Ll(a,b,t)*(phi[i+1] - 0)
                dphi[i] = 1/h**2*1j/Ls(a,b,t,omega)**2*(phi[i+1]-2*phi[i]+ 0 ) + 1/2*(i+1)*dLs(b,t,omega)/Ls(a,b,t,omega)*(phi[i+1] - 0)

                
            if  i == r-1:
                # print(i,phi[i])
                
                dphi[i] = 1/h**2*1j/Ls(a,b,t,omega)**2*( 0 - 2*phi[i]+ phi[i-1] ) + 1/2*(i+1)*dLs(b,t,omega)/Ls(a,b,t,omega)*(0 - phi[i-1])                
                # print('Calculado!')
                
    # print(dphi)
    return dphi



try:
    a = 1

    # =============================================================================
    # # Casos específicos
    # =============================================================================
    # CASO 1 Pared de movimiento lineal
    #   A)  a = 1, b = 3, tf = 1
    #   B) a = 1, b = 1, t = 1        
    #   C) a = 1, b = 0.2, t = 1  
    
    # CASO 4 Pared de movimiento lineal
    #   A)  a = 1, b = 2, tf = 5
    #   B) a = 1, b = 2, tf = 50        
    if caso == 0:     
        abc = int(input('¿Qué variante calcular? (1-5) \n'))
        
        if abc not in np.arange(1,6):
            caso = -1
        else:
            tf = 1
            b = 2
            if abc == 1: 
                b = 3
                s = 'A'
            if abc == 2: 
                b = 1
                s = 'B'
            if abc == 3:
                b = 0.2
                s = 'C'
            if abc == 4:
                tf = 5
                s = 'D'
            if abc == 5:
                tf = 50
                s = 'E'
                
    # CASO 2 Pared de movimiento sinusoidal
    #   A) a = 1, b = 0.6, w = 2, tf = 3
    #   B) a = 1, b = 0.4, w = 2, tf = 3        
    #   C) a = 1, b = 0.2, w = 2, tf = 3  
    
    # CASO 3 Pared de movimiento sinusoidal
    #   A) a = 1, b = 0.2, w = 10, tf = 3
    #   B) a = 1, b = 0.2, w = 5, tf = 3              
    #   C) a = 1, b = 0.2, w = 2, tf = 3  
                
    if caso == 1:
        abc = int(input('¿Qué variante calcular? (1-6) \n'))
        
        if abc not in np.arange(1,7):
            caso = -1
        else:
            tf = 3
            b = 0.2
            if abc == 1: 
                b = 0.6
                s = 'A'
            if abc == 2: 
                b = 0.4
                s = 'B'
            if abc == 3:
                b = 0.2
                s = 'C'
            if abc == 4:
                w = 10
                s = 'D'
            if abc == 5:
                w = 5
                s = 'E'
            if abc == 6:
                w = 2
                s = 'F'
    
    # =============================================================================
    # # Resolución numérica
    #  Esto se hace con ivp
    # =============================================================================
    
    h = 1/m
    z = np.linspace(z0+h,zf-h,m-2)
    # z = np.transpose(z)
    # print('tamaño z  =  ', len(z))
    
    y0 = f(z)
    # y0[0], y0[-1] = 0,0
    
    t = np.linspace(t0,tf,Nt)
    
      
    n = 0
    
    if caso == 0:
        tp1 = time.time()
        sol = ivp(resl,(t0,tf), y0, t_eval=t,)
        tp2 = time.time()
        
        tp = tp2-tp1
        print('Tiempo que tarda con njit: \n','- Lineal:= ', tp)
        
        if sol.status == 0:
            resl = np.zeros((m,Nt))
            
            resl[1:-1,:] = sol.y
            
            resl = np.transpose(resl)

            
            aux = nombre1.split('.')
            nombre3 = aux[0] + ' ' + s + '.' + aux[1]
            direc = 'Animaciones pycharm4'
                
            with open(nombre3, 'w') as f:    
                f.write(f'{Nt} ')
                # print(tf)
                f.write(f'{m} ')
                f.write(f'{tf} ')
                                
                for i in range(Nt):
                    for j in range(m):
                        f.write(f'{resl[i,j]} ')
                        n += 1
                        
            dirnom = direc + '/' + nombre1 
            sh.copy(nombre3, dirnom)
            
        else:
            print('\n \033[31mNo hay resultado para la ecuación lineal.\033[0m')
            print(sol.message)
            


    
    if caso  == 1:    
        tp1 = time.time()
        sol2 = ivp(resc,(t0,tf), y0, t_eval=t,)
        tp2 = time.time()
        
        tp3 = tp2-tp1
        print('Tiempo que tarda con njit: \n',' - Circular:= ',tp3)
        
        if sol2.status == 0:

            resc = np.zeros((m,Nt))
            
            resc[1:-1,:] = sol2.y
        
            resc = np.transpose(resc)
    
            aux = nombre2.split('.')
            nombre4 = aux[0] + ' ' + s + '.' + aux[1]
            direc = 'Animaciones pycharm4'
    
    
            with open(nombre4, 'w') as f:  
                f.write(f'{Nt} ')
                print(m)
                f.write(f'{m} ')
                f.write(f'{tf} ')
            
                for i in range(Nt):
                    for j in range(m):
                        f.write(f'{resc[i,j]} ')
                        n += 1
    
            dirnom = direc + '/' + nombre2
            sh.copy(nombre2, dirnom)
            
        else:
            print('\n \033[31mNo hay resultado para la ecuación circular.\033[0m', )
            print(sol2.message)

    
    print(f'\n \033[32mSe han enviado {n} datos al fichero\033[0m')
    
  
except caso > 1: 
    print('\033[31mNo existe tal caso, prueba de nuevo un caso válido.\033[0m')
except caso < 0:
    print('La variante elegida no es válida o no existe, inténtalo de nuevo.')
except ValueError:
    print('Inserta un carácter válido')
    
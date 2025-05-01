#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 12:55:48 2025

@author: isaac
"""




import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from matplotlib import rc
import netCDF4 as nc
from numba import njit, prange
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'  # or 'DejaVu Serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
#%%

# =============================================================================
# Functions to be called in the primary code below
# ============================================================================

def grid_maker (x,y,r,c):
    solving_grid = np.ones((len(x),len(y)))
    for pos in range (0,len(c[:,0])):
        for x_int in range (0,len(x)):
            for y_int in range (0,len(y)):
                 position = np.sqrt((x[x_int]-c[pos,0])**2+((y[y_int]-c[pos,1])**2))
                 if position <r:
                     solving_grid[x_int,y_int] = 0
                     
        
    Ghost_cells = np.zeros((len(x),len(y)))
    Num_Ghost = 0
    for i in range(1,Nx-1):
        for j in range (1,Ny-1):
            Solid= solving_grid[i,j] == 0
            FN = solving_grid[i+1,j]==1 or solving_grid[i+1,j]==1 or solving_grid[i+1,j]==1 or solving_grid[i+1,j]==1
            Ghost_cells[i,j]= Solid and FN
            Num_Ghost=Num_Ghost+int(Ghost_cells[i,j])
            
    Ghost_array = np.zeros((2,Num_Ghost),dtype=int)
    gn = 0
    for i in range(1,Nx-1):
        for j in range (1,Ny-1):
            if Ghost_cells[i,j] ==1:
                Ghost_array[0,gn]= i
                Ghost_array[1,gn] =j
                gn = gn +1

    return(solving_grid, Ghost_array,Ghost_cells, Num_Ghost) 

def interpolator (Ui,color):
    """
    Interpolates cell center values to cell face values
    
    Args:
        Ui: cell center velocites 
      
    Returns:
        Cell Face Velocities
    """    
    Big_U = np.zeros_like(Ui) 
    int_u_array = Ui.copy()       
    for yi_int in range (0,Ny):
        for xi_int in range (0,Nx-1):
            P = color[xi_int,yi_int]*color[xi_int+1,yi_int]
            Big_U[0,xi_int,yi_int] = P*(int_u_array[0,xi_int,yi_int] + int_u_array[0,xi_int+1,yi_int])/2
            
    for yi_int in range (0,Ny-1):
        for xi_int in range (0,Nx):
            P = color[xi_int,yi_int]*color[xi_int,yi_int+1]
            Big_U[1,xi_int,yi_int] = P*(int_u_array[1,xi_int,yi_int] + int_u_array[1,xi_int,yi_int+1])/2
           
    return Big_U


def Dir_BC (Ud):
    """
    Function applies the Dirichlet Boundary Conditions to the input array, such 
    that the interpolated value between cell centers of the fluid and ghost cells
    on the boundary will equal the prescribed boudnary condition on the wall, which 
    is located on a cell face between the fluid and ghost cells. 
    This only prescribes bonudary conditions on the edges of the domain, and not 
    for an imbedded body. 
    
    Args:
        Ud: cell center values 
      
    Returns:
        Modified cell center velocties that comply with given boudary conditions
    """     
    
    U_BC = Ud.copy()
    
    U_BC[0,0,:] =  2-Ud[0,1,:] #Left #u=1 inflow
    U_BC[0,-1,:] = Ud[0,-2,:]  #Right #du/dx =0
    U_BC[0,:,0] =  Ud[0,:,1]   #Bottom #Periodic
    U_BC[0,:,-1] = Ud[0,:,-2] #Top #Periodic
    
    U_BC[1,0,:] =  -Ud[1,1,:]  #Left v=0
    U_BC[1,-1,:] = Ud[1,-2,:] #Right dv/dy=0
    U_BC[1,:,0] =  Ud[1,:,1]  #Bottom v=0
    U_BC[1,:,-1] = Ud[1,:,-2] #Top v=0
    
    return (U_BC)
    
         
def Neu_BC (P_array):
    """
    Function applies the Neumann Boundary Conditions to the input array, such 
    that the derivative between the fluid and ghost cell center values are 0
    
    Args:
        Ud: cell center values 
      
    Returns:
        Modified cell center velocties that comply with given boudary conditions
    """   
    
    BC_array=P_array.copy()
    
    BC_array[-1,:] = BC_array[-2,:]
    BC_array[0,:] = BC_array[1,:]
    BC_array[:,-1] = BC_array[:,-2]
    BC_array[:,0] = BC_array[:,1]
    
    return (BC_array)

def GC_Dir_BC (Val_array, GhostArray, Num_Ghost,Color):
    for gn in range (0,Num_Ghost):
        i = int(GhostArray[0,gn])
        j = int(GhostArray[1,gn])
        count = (Color[i+1,j] + Color[i+1,j] + Color[i+1,j] + Color[i+1,j])
        average = ((Color[i+1,j])*Val_array[i+1,j]) + ((Color[i-1,j])*Val_array[i-1,j]) + ((Color[i,j+1])*Val_array[i,j+1]) + ((Color[i,j-1])*Val_array[i,j-1])
        Val_array[i,j] = -average / count # DIR
    return(Val_array)

def GC_Neum_BC (Val_array, GhostArray, Num_Ghost,Color):
    """
    Neumann BC - Ghost Cell 
    
    
    """   
    for gn in range (0,Num_Ghost):
        i = int(GhostArray[0,gn])
        j = int(GhostArray[1,gn])
        count = (Color[i+1,j] + Color[i+1,j] + Color[i+1,j] + Color[i+1,j])
        average = ((Color[i+1,j])*Val_array[i+1,j]) + ((Color[i-1,j])*Val_array[i-1,j]) + ((Color[i,j+1])*Val_array[i,j+1]) + ((Color[i,j-1])*Val_array[i,j-1])
        Val_array[i,j] = average/count # NUM 
    return(Val_array)


def Gradient (Up,dx,dy,Nx,Ny,Color):
    """
    Calculates the gradient of a given array using central differencing
    
    Args:
        Up: Cell center values 
        dx: Delta x value (cell center spacing)
        dy: Delta y value (cell center spacing)
        Nx: Number of x grid points
        Ny: Number of y grid points
      
    Returns:
        2-d Gradient array for x and y gradients
    """   
    Grad = np.zeros((2,Nx,Ny))
    for xg_int in range (1,Nx-1):
        for yg_int in range (1,Ny-1):
            P = Color[xg_int,yg_int]
            
            # P+1(C+1) + Pi(Ci-1 - Ci+1) - Pi-1(Ci-1)
            
            Grad[0,xg_int,yg_int] = P* ((Up[xg_int+1,yg_int]*Color[xg_int+1,yg_int])+ (Up[xg_int,yg_int]*(Color[xg_int-1,yg_int]-Color[xg_int+1,yg_int]))-(Up[xg_int-1,yg_int]*Color[xg_int-1,yg_int])      ) /(2*dx)
            #P = Color[xg_int,yg_int-1]*Color[xg_int,yg_int+1]
            Grad[1,xg_int,yg_int] = P* ((Up[xg_int,yg_int+1]*Color[xg_int,yg_int+1])+ (Up[xg_int,yg_int]*(Color[xg_int,yg_int-1]-Color[xg_int,yg_int+1]))-(Up[xg_int,yg_int-1]*Color[xg_int,yg_int-1])      ) /(2*dy)
            #Grad[1,xg_int,yg_int] = P*(Up[xg_int,yg_int+1] - Up[xg_int,yg_int-1])/(2*dy)
    return Grad

def ForwardGradient (Up,dx,dy,Nx,Ny,Color):
    """
    Calculates the gradient of a given array using a forward differencing scheme. 
    This is used to update the cell face values
    
    Args:
        Up: Cell center values 
        dx: Delta x value (cell center spacing)
        dy: Delta y value (cell center spacing)
        Nx: Number of x grid points
        Ny: Number of y grid points
      
    Returns:
        2-d Gradient array for x and y gradients
    """   
    Grad = np.zeros((2,Nx,Ny))
    for xg_int in range (1,Nx-1):
        for yg_int in range (1,Ny-1):
            P = Color[xg_int,yg_int]*Color[xg_int+1,yg_int]
            Grad[0,xg_int,yg_int] = P*(Up[xg_int+1,yg_int] - Up[xg_int,yg_int])/(dx)
            P = Color[xg_int,yg_int]*Color[xg_int,yg_int+1]
            Grad[1,xg_int,yg_int] = P*(Up[xg_int,yg_int+1] - Up[xg_int,yg_int])/(dy)
    return Grad
  
def Divergence (U,Nx,Ny,dx,dy):
    """
    Calculates the divergence of a given array
    
    Args:
        Up: Cell center values 
        Nx: Number of x grid points
        Ny: Number of y grid points
        dx: Delta x value (cell center spacing)
        dy: Delta y value (cell center spacing)
        
      
    Returns:
        Array of Divergence values
    """   
    
    Val = U.copy()
    Div = np.zeros((Nx,Ny))
    for x_int in range(1,Nx-1):
        for y_int in range(1,Ny-1):
            Div[x_int,y_int] = (((Val[0,x_int,y_int] - Val[0,x_int-1,y_int])/(dx))
                                +((Val[1,x_int,y_int] - Val[1,x_int,y_int-1])/(dy)) )  
            
            
    return(Div)


def Advec_Diffuse_RHS (U_n,CU_n,U_n1,CU_n1,dx,dy,Nx,Ny,Visco,dt,Color):
    """
    Calculates Right hand side of the Advection Diffusion equation given in step 1 
    of the fractional step method
    
    Args:
        U_n: Cell center velocites at current time step (n)
        CU_n: Cell face velocites at current time step (n)
        U_n1: Cell center velocites at previous time step (n-1)
        CU_n1: Cell face velocites at previous time step (n-1)
        dx: Delta x value (cell center spacing)
        dy: Delta y value (cell center spacing)
        Nx: Number of x grid points
        Ny: Number of y grid points
        Visco = prescribed viscosity (1/Re)
        dt = Delta t value 
        
    Returns:
        Array of values of the RHS of the AD equation to be inputs for the Gauss Sidel solver
    """   
        
    U_star_RHS = np.zeros_like(U_n)   
    for i in range (0,2):
        for adx_int in range (1,Nx-1):
            for ady_int in range(1,Ny-1): 
                
                ### Color Function Coeffeicents 
                P = Color[adx_int,ady_int]
                N = Color[adx_int,ady_int+1]
                E = Color[adx_int+1,ady_int]
                S = Color[adx_int,ady_int-1]
                W = Color[adx_int-1,ady_int]
                
                ### Explicit Diffusion Term
                Duw = (W)*((U_n[i,adx_int-1,ady_int]))
                Due = (E)*((U_n[i,adx_int+1,ady_int]))
                Dun = (N)*((U_n[i,adx_int,ady_int+1]))
                Dus = (S)*((U_n[i,adx_int,ady_int-1]))
                Dupx = ((-2*P) - (1-W) - (1-E))*U_n[i,adx_int,ady_int]
                Dupy = ((-2*P) - (1-N) - (1-S))*U_n[i,adx_int,ady_int]
                
                
                
                Dif_x = ((Due+Dupx+Duw)/(dx**2))
                Dif_y = ((Dun+Dupy+Dus)/(dy**2))
                
                Dif = (dt*0.5*Visco) * (Dif_x+Dif_y)
              
                ### Convective Term t=n
                uw = (W)*((U_n[i,adx_int-1,ady_int]+U_n[i,adx_int,ady_int])/2)
                ue = (E)*((U_n[i,adx_int+1,ady_int]+U_n[i,adx_int,ady_int])/2)
                un = (N)*((U_n[i,adx_int,ady_int+1]+U_n[i,adx_int,ady_int])/2)
                us = (S)*((U_n[i,adx_int,ady_int-1]+U_n[i,adx_int,ady_int])/2)
                
                Con1 = -(3/2)*dt * (
                    ( ( (CU_n[0,adx_int,ady_int]   *(ue)) - 
                        (CU_n[0,adx_int-1,ady_int] *(uw))) /(dx))
                    
                   +( ( (CU_n[1,adx_int,ady_int]   *(un)) - 
                        (CU_n[1,adx_int,ady_int-1] *(us))) /(dy))  )
                
                ### Convective Term t=n-1
                uw1 = (W)*((U_n1[i,adx_int-1,ady_int]+U_n1[i,adx_int,ady_int])/2)
                ue1 = (E)*((U_n1[i,adx_int+1,ady_int]+U_n1[i,adx_int,ady_int])/2)
                un1 = (N)*((U_n1[i,adx_int,ady_int+1]+U_n1[i,adx_int,ady_int])/2)
                us1 = (S)*((U_n1[i,adx_int,ady_int-1]+U_n1[i,adx_int,ady_int])/2)
                
                Con2 = (1/2)*dt * (
                    ( ( (CU_n1[0,adx_int,ady_int]   *(ue1)) - 
                        (CU_n1[0,adx_int-1,ady_int] *(uw1))) /(dx))
                    
                   +( ( (CU_n1[1,adx_int,ady_int]   *(un1)) - 
                        (CU_n1[1,adx_int,ady_int-1] *(us1))) /(dy))  )
                
           
                U_star_RHS[i,adx_int,ady_int] = (U_n[i,adx_int,ady_int] + Dif + Con1 +Con2)#- dt*Grad_P[i,adx_int,ady_int]

    return U_star_RHS

def Unsteady_Source_GaussSidel (Array,Source,Nx,Ny,dt,dx,Visco,Color,Ghost_array, NumGhost, omega=1.7):
    
    """
    Calculates u* values using Gauss-Sidel 
    
    Args:
        Array: Cell center velocites at current time step (n) to be used as an inital guess
        Source: RHS of the advection diffusion equation solved in the function above
        Nx: Number of x grid points
        Ny: Number of y grid points
        dt = Delta t value 
        dx: Delta x value (cell center spacing, assuming evenly spaced grid)
        Visco = prescribed viscosity (1/Re)
        Omega = Relaxation factor for SOR GS (default = 1.7)
        
    Returns:
        Array of values of for u*
    """   
    
    
   
    
    k = 10000                   ## Maximum number of iterations 
    threshold = 10**-7        ## Threshold for convergence
    G_solutions = Array.copy()
    Source = Source.copy()
    
    G_Rcalc = np.zeros((2,Nx,Ny))
    G_Rxvalues= []
    G_Ryvalues= []
    
    G_Convergence = False
    alpha = (1+((2*dt*Visco)/(dx**2)))

    for iter_val in range (0,k):
        if G_Convergence == False:
            ## Calcuates updated values for u (i=0), and v (i=1)
            for i in range (0,2): 
                for UGx_int in range (1,Nx-1):
                    for UGy_int in range (1,Ny-1):
                        
                        P =   Color[UGx_int,UGy_int]
                        N = Color[UGx_int,UGy_int+1]
                        E = Color[UGx_int+1,UGy_int]
                        S = Color[UGx_int,UGy_int-1]
                        W = Color[UGx_int-1,UGy_int]
                       
                        Dup = (1/((alpha) + (((dt*Visco)/(2*(dx**2)))*(4-W-E-N-S))  )  )
                        if ~np.isinf(Dup): 
                            Dup = Dup
                        else:
                            Dup=0
                        
                        Temp_Avg = (((0.5*Visco*dt/dx**2))*
                                   ((W)*G_solutions[i,UGx_int-1,UGy_int] +
                                    (S)*G_solutions[i,UGx_int,UGy_int-1]+
                                    (E)*G_solutions[i,UGx_int+1,UGy_int]+
                                    (N)*G_solutions[i,UGx_int,UGy_int+1]))
                      
                        Temp = P*(Dup*((Source[i,UGx_int,UGy_int]) + Temp_Avg))# + ((1-P)*G_solutions[i,UGx_int,UGy_int]   )
                        
                        ## applies relaxation (not applied if omega=1.0)
                        G_solutions[i,UGx_int,UGy_int] = ((omega * Temp) + 
                                    (1-omega)*G_solutions[i,UGx_int,UGy_int])
              
            ## Ensures Dirichlet Boundary Conditions are met
            G_solutions[:,:,:] = Dir_BC(G_solutions[:,:,:])
            G_solutions[0,:,:] = GC_Dir_BC (G_solutions[0,:,:], Ghost_array, NumGhost,Color)
            G_solutions[1,:,:] = GC_Dir_BC (G_solutions[1,:,:], Ghost_array, NumGhost,Color)
            
            ## Calcuates residual values for u (i=0), and v (i=1)
            for i in range (0,2):         
                for UGx_int in range (1,Nx-1):
                   for UGy_int in range (1,Ny-1):
                    
                    
                    P =   Color[UGx_int,UGy_int]
                    N = Color[UGx_int,UGy_int+1]
                    E = Color[UGx_int+1,UGy_int]
                    S = Color[UGx_int,UGy_int-1]
                    W = Color[UGx_int-1,UGy_int]
                   
                    Dup = (1/((alpha) + (((dt*Visco)/(2*(dx**2)))*(4-W-E-N-S))  )  )
                    if ~np.isinf(Dup): 
                        Dup = Dup
                    else:
                        Dup=0
                    
                    Temp_Avg = (((0.5*Visco*dt/dx**2))*
                               ((W)*G_solutions[i,UGx_int-1,UGy_int] +
                                (S)*G_solutions[i,UGx_int,UGy_int-1]+
                                (E)*G_solutions[i,UGx_int+1,UGy_int]+
                                (N)*G_solutions[i,UGx_int,UGy_int+1]))
                  
                    Temp = P*(Dup*((Source[i,UGx_int,UGy_int]) + Temp_Avg))# + ((1-P)*G_solutions[i,UGx_int,UGy_int]   )
                    
                    G_Rcalc[i,UGx_int,UGy_int] =P*(G_solutions[i,UGx_int,UGy_int] - Temp)
                   
            G_Rx = np.sqrt((np.average(G_Rcalc[0,:,:])**2))
            G_Ry = np.sqrt((np.average(G_Rcalc[1,:,:])**2))
            G_Rxvalues.append(G_Rx)
            G_Ryvalues.append(G_Ry)
            
            ## The following ensures both u and v have reached convergence
            
            if G_Rx > threshold and G_Ry > threshold: 
                G_Convergence = False
            if G_Rx < threshold and G_Ry < threshold:
                G_Convergence = True
     
                
        if G_Convergence == True:            
            break 
    return(G_solutions)




def Source_GaussSidel (Array,Source,Nx,Ny,dx,Color,Ghost_array, NumGhost, omega=1.7):
    
    """
    Solves the Pressure-Poisson equation for p n+1 values using Gauss-Sidel  
    
    Args:
        Array: Cell center presssures at current time step (n) to be used as an inital guess
        Source: RHS of the PPE (Divergence of cell faces)
        Nx: Number of x grid points
        Ny: Number of y grid points
        dx: Delta x value (cell center spacing, assuming evenly spaced grid)
        Omega = Relaxation factor for SOR GS (default = 1.7)
        
    Returns:
        Array of values of for p n+1
    """   
    
    
    
    k = 10000                   ## Maximum number of iterations 
    threshold = 10**-7          ## Threshold for convergence
    
    G_solutions=Array.copy()
    S_array =   Source.copy()
    G_Rcalc = np.zeros((Nx,Ny))
    G_Rvalues= []
   
    G_Convergence = False
    for iter_val in (range (0,k)):
        if G_Convergence == False:
            
            ## Calcuates updated values
            for Gx_int in range (1,Nx-1):
                for Gy_int in range (1,Ny-1):
                    
                    
                    P = Color[Gx_int,Gy_int]
                    N = Color[Gx_int,Gy_int+1]
                    E = Color[Gx_int+1,Gy_int]
                    S = Color[Gx_int,Gy_int-1]
                    W = Color[Gx_int-1,Gy_int]
                    
                    
                    Dup = 1/(((-4) + (1-W) +(1-N)+ (1-E)+ (1-S)))
                    
                    if ~np.isinf(Dup): 
                        Dup = Dup
                    else:
                        Dup=0
                    
                   # Dup = (1/((P*4)+(1-N)+(1-E)+(1-W)+(1-S)))
                    
                    Temp_U = ((N*G_solutions[Gx_int,Gy_int+1])+
                              (W*G_solutions[Gx_int-1,Gy_int])+
                              (E*G_solutions[Gx_int+1,Gy_int])+
                              (S*G_solutions[Gx_int,Gy_int-1]))    
                    U = Dup*(-Temp_U + ((dx**2)*S_array[Gx_int,Gy_int]))#+ ((1-P)*G_solutions[Gx_int,Gy_int])
                    ## applies relaxation (not applied if omega=1.0)
                    smooth_U = ((omega * U) + 
                                (1-omega)*G_solutions[Gx_int,Gy_int])
                    
                    G_solutions[Gx_int,Gy_int] = smooth_U
                    
            ## Ensures Neumann Boundary Conditions are met
            G_solutions[:,:] = Neu_BC(G_solutions[:,:]) 
            G_solutions[:,:] = GC_Neum_BC (G_solutions[:,:], Ghost_array, NumGhost,Color)
            ## Calcuates residual
            for Gx_int in range (1,Nx-1):
               for Gy_int in range (1,Ny-1):
                   
                   P = Color[Gx_int,Gy_int]
                   N = Color[Gx_int,Gy_int+1]
                   E = Color[Gx_int+1,Gy_int]
                   S = Color[Gx_int,Gy_int-1]
                   W = Color[Gx_int-1,Gy_int]
                   
                   Dup = 1/(((-4) +(1-W) + (1-N)+ (1-E)+ (1-S)))
                   if ~np.isinf(Dup): 
                       Dup = Dup
                   else:
                       Dup=0
                   Temp_U = ((N*G_solutions[Gx_int,Gy_int+1])+
                             (W*G_solutions[Gx_int-1,Gy_int])+
                             (E*G_solutions[Gx_int+1,Gy_int])+
                             (S*G_solutions[Gx_int,Gy_int-1]))    
                   U = Dup*(-Temp_U + ((dx**2)*S_array[Gx_int,Gy_int])) #+ ((1-P)*G_solutions[Gx_int,Gy_int])
                  
                   
                   G_Rcalc[Gx_int,Gy_int] = P*(G_solutions[Gx_int,Gy_int] - U)   #/ (dx**2)
                   
            G_R = np.sqrt((np.average((G_Rcalc[:])**2)))
            #print(G_R)
            G_Rvalues.append(G_R)
            #print(G_R)
            if G_R > threshold:
                G_Convergence = False
                
            if G_R < threshold:
                
                G_Convergence = True                
        if G_Convergence == True:
            break 
   
    return(G_solutions[:,:])
# =============================================================================
# @njit(parallel=True)
# def update_GaussSeidel(G_solutions, S_array, dx, omega, Nx, Ny):
#     for Gx_int in prange(1, Nx - 1):
#         for Gy_int in range(1, Ny - 1):
#             Temp_U = (
#                 G_solutions[Gx_int, Gy_int + 1] +
#                 G_solutions[Gx_int - 1, Gy_int] +
#                 G_solutions[Gx_int + 1, Gy_int] +
#                 G_solutions[Gx_int, Gy_int - 1]
#             )
#             U = 0.25 * (Temp_U - (dx**2) * S_array[Gx_int, Gy_int])
#             smooth_U = omega * U + (1 - omega) * G_solutions[Gx_int, Gy_int]
#             G_solutions[Gx_int, Gy_int] = smooth_U
# 
# @njit(parallel=True)
# def calculate_residual(G_solutions, S_array, G_Rcalc, dx, Nx, Ny):
#     for Gx_int in prange(1, Nx - 1):
#         for Gy_int in range(1, Ny - 1):
#             Temp_U = (
#                 G_solutions[Gx_int, Gy_int + 1] +
#                 G_solutions[Gx_int - 1, Gy_int] +
#                 G_solutions[Gx_int + 1, Gy_int] +
#                 G_solutions[Gx_int, Gy_int - 1]
#             )
#             U = 0.25 * (Temp_U - (dx**2) * S_array[Gx_int, Gy_int])
#             G_Rcalc[Gx_int, Gy_int] = G_solutions[Gx_int, Gy_int] - U
# 
# def Source_GaussSidel(Array, Source, Nx, Ny, dx, omega=1.7):
#     k = 10000
#     threshold = 1e-7
# 
#     G_solutions = Array.copy()
#     S_array = Source.copy()
#     G_Rcalc = np.zeros((Nx, Ny))
# 
#     for iter_val in range(k):
#         update_GaussSeidel(G_solutions, S_array, dx, omega, Nx, Ny)
# 
#         # Apply Neumann boundary condition — assumed to be outside numba
#         G_solutions[:,:] = Neu_BC(G_solutions[:,:])
# 
#         calculate_residual(G_solutions, S_array, G_Rcalc, dx, Nx, Ny)
#         G_R = np.sqrt(np.mean(G_Rcalc**2))
# 
#         if G_R < threshold:
#             break
# 
#     return G_solutions
# =============================================================================

def FlowCorrection (Uf,Ny,dy):
    Face_Velocities = Uf.copy()
    Dif = (np.sum((Face_Velocities[0,1:-1] - Face_Velocities[-2,1:-1])*dy))/(dy*(Ny-2))
    #print(Dif)
    Face_Velocities[-2,1:-1]=(Face_Velocities[-2,1:-1]+Dif)
 
    #Dif = (np.sum((Face_Velocities[0,:] - Face_Velocities[-2,:])*dy))/(dy*Ny)
    #print(Dif)

    return(Face_Velocities)
    

#%%

# =============================================================================
# Constants set up outside primary loop. The primary loop solves for different
# dx values given in the Ds array
# ============================================================================
#dx= 0.26 0.4
Main_Threshold = 10**-8            ## threshold for convergence to end primarily loop 
Ds_array = [1/10]  ## Dx values to test
Re = 150   # 1/100 #Reynolds  Number
V =  1/Re           #Viscocity =1/Re
radius = 0.5
save = False         #Saves the solved values for plotting in a seperate code
Max_ds = Ds_array[-1]
X_length = 20
Y_length  = 6
max_x_array = np.arange(-Max_ds/2,(X_length*radius)+(2*Max_ds/2),Max_ds)
max_y_array = np.arange(-Max_ds/2,(Y_length*radius)+(2*Max_ds/2),Max_ds)
center = np.array([[6*radius,3*radius]])


#max_x_array = np.arange(-Max_ds/2,2+(2*Max_ds/2),Max_ds)
#max_y_array = np.arange(-Max_ds/2,1+(2*Max_ds/2),Max_ds)


U_array_full =np.zeros((len(Ds_array),len(max_x_array),len(max_y_array)))
V_array_full =np.zeros((len(Ds_array),len(max_x_array),len(max_y_array)))
UF_array_full =np.zeros((len(Ds_array),len(max_x_array),len(max_y_array)))
VF_array_full =np.zeros((len(Ds_array),len(max_x_array),len(max_y_array)))
Pressure_full =np.zeros((len(Ds_array),len(max_x_array),len(max_y_array)))
Index = np.zeros((len(Ds_array))) ## Index values used to index the arrays above for graphing


#%%
for s_int in range (0,len(Ds_array)):
   
    # ============================================================================
    # Constants and loop specific arrays
    
    dS=Ds_array[s_int]
    
    x_array = np.arange(-dS/2,(X_length*radius)+(2*dS/2),dS)
    y_array = np.arange(-dS/2,(Y_length*radius)+(2*dS/2),dS)
    
    dt = dS/2                       # dt to comply with CFL condition
    t_array = np.arange(0,100+dt,dt) # Maximum time (number of iterations)
    
    #x_array = np.arange(-dS/2,2+(2*dS/2),dS) #Set up to have 1/dx fluid cells + 2 ghost cells
    Nx = len(x_array)
    #y_array = np.arange(-dS/2,1+(2*dS/2),dS)
    Ny = len(y_array)
   
    #color= np.zeros((Nx,Ny))
    #color[:,:]= 1
    
    color, Ghost_array,Ghost_cells, Num_Ghost = grid_maker(x_array, y_array, radius, center)
    
    plt.figure(figsize=(X_length,Y_length))
    rc('font',weight='normal',size=30)
    plt.xticks(x_array, minor=True)
    plt.yticks(y_array, minor=True)
    #plt.minorticks_on()
    plt.grid(which='both')
    plt.pcolormesh(x_array,y_array,color[:,:].transpose(),cmap='binary_r') 
    plt.colorbar()
    plt.show()  
    
    # ============================================================================
    #Empty arrays to store values 
    
    # u and v values stored as 0 and 1, respectively, in the first index
    U_sol = np.zeros((2,len(t_array),Nx,Ny))
    Con_U_sol = np.zeros((2,len(t_array),Nx,Ny))
    P_sol = np.zeros((len(t_array),Nx,Ny))
    
    
    U_sol[0,0,0,:] = 1
    ## Apply Dirichlet BC
    U_sol[:,0,:,:] = Dir_BC(U_sol[:,0,:,:])

    U_sol[0,1,:,:],U_sol[1,1,:,:] = U_sol[0,0,:,:],U_sol[1,0,:,:]
    ## Interpolate Cell centers to faces
    Con_U_sol[:,0,:,:] = interpolator(U_sol[:,0,:,:],color)
    Con_U_sol[0,1,:,:],Con_U_sol[1,1,:,:] = Con_U_sol[0,0,:,:],Con_U_sol[1,0,:,:]

    
    # ============================================================================
    # After setting initial conditions, begin to march forward in time solving the 
    # partial step method for n+1 solutions of u, U and p
    
    for t_int in tqdm(range (2,len(t_array))): #3)): #
    
        ### First solve the RHS of eq. #. This uses the above Advection Diffusion function
        U_RHS = Advec_Diffuse_RHS(U_sol[:,t_int-1,:,:],Con_U_sol[:,t_int-1,:,:],U_sol[:,t_int-2,:,:],Con_U_sol[:,t_int-2,:,:],dS,dS,Nx,Ny,V,dt,color)   
        
        ### Then use the RHS to iteratively solve for u*
        U_Star = Unsteady_Source_GaussSidel(U_sol[:,t_int-1,:,:], U_RHS, Nx, Ny, dt, dS, V,color,Ghost_array, Num_Ghost)
        
        ### Interpolate u* to get * Face velocities (U*)
        PreCorrection_U = interpolator(U_Star,color)
        
        U_Correction = FlowCorrection(PreCorrection_U[0,:,:],Ny,dS)
        Con_U_Star = PreCorrection_U.copy()
        Con_U_Star[0,:,:] = U_Correction
        
        
        ## Calculate the divergence of U* 
        Div_array = (1/dt)*Divergence(Con_U_Star, Nx, Ny, dS, dS)
        #print(np.sum(Div_array))
 
        ## Ensure that the sum of the divergence is 0 (or very small at least)
        
        if abs(np.sum(Div_array[1:-1,1:-1]))>0.1:
            print("Divergence Criteria Not Met")
            print("Divergence =",np.sum(Div_array))
            Break_index = t_int-1         
            break
        
        else:
            
        ##Solve the Pressure Poisson equation using the Divergence of U*
            P_n  = Source_GaussSidel(P_sol[t_int-1,:,:], Div_array, Nx, Ny, dS,color,Ghost_array, Num_Ghost)
        ##Calculate the gradient of pressure for cell centeres and faces
            Grad_P = Gradient(P_n, dS, dS, Nx, Ny,color)
           # Interp_p = simple_int(P_n,Nx,Ny)
            Face_Grad_P = ForwardGradient(P_n, dS, dS, Nx, Ny,color)

        ##Update the velocities
            New_U = np.zeros_like(U_Star)
            New_CU = np.zeros_like(Con_U_Star)
            for ux_int in range (0,Nx):
                for uy_int in range (0,Ny):
                    for i in range (0,2):
                        P  =color[ux_int,uy_int]
                        New_U[i,ux_int,uy_int] =  P*  (  U_Star[i,ux_int,uy_int] - (dt*Grad_P[i,ux_int,uy_int]))
                        New_CU[i,ux_int,uy_int] = P* (Con_U_Star[i,ux_int,uy_int] - (dt*Face_Grad_P[i,ux_int,uy_int]))
        ## Fill arrays inside the primary dx loop 
            U_sol[:,t_int,:,:] = Dir_BC(New_U)
            Con_U_sol[:,t_int,:,:] = New_CU
            P_sol[t_int,:,:] =P_n
            
        ## Calculate residual of the cell center velocities (u and v)            
            Residual = np.sqrt(np.average((P_sol[t_int,:,:]-U_sol[:,t_int-1,:,:])**2))
            if Residual < Main_Threshold:
                Break_index = t_int
                break
            if t_int == len(t_array)-1:
                Break_index = t_int
                #%%
            plt.figure(figsize=(X_length,Y_length))
            rc('font',weight='normal',size=25) 
            clevs = np.arange(-150,150,0.1)
            plt.contourf(x_array[:],y_array[:],P_sol[t_int,:,:].transpose(),cmap='coolwarm',levels=clevs,extend='both')#vmin=0,vmax=1)
            plt.title("p")
            plt.colorbar()
            plt.xlim(x_array[0],x_array[-1])
            plt.ylim(y_array[0],y_array[-1])
            plt.close()
            
            plt.figure(figsize=(X_length,Y_length))
            rc('font',weight='normal',size=25) 
            clevs = np.arange(-15,15,0.1)
            plt.contourf(x_array[:],y_array[:],Grad_P[0,:,:].transpose(),cmap='coolwarm',levels=clevs,extend='both')#vmin=0,vmax=1)
            plt.title("dp/dx")
            plt.colorbar()
            plt.xlim(x_array[0],x_array[-1])
            plt.ylim(y_array[0],y_array[-1])
            plt.close()
            
            plt.figure(figsize=(X_length,Y_length))
            rc('font',weight='normal',size=25) 
            clevs = np.arange(-15,15,0.1)
            plt.contourf(x_array[:],y_array[:],Grad_P[1,:,:].transpose(),cmap='coolwarm',levels=clevs,extend='both')#vmin=0,vmax=1)
            plt.title("dp/dy")
            plt.colorbar()
            plt.xlim(x_array[0],x_array[-1])
            plt.ylim(y_array[0],y_array[-1])
            plt.close()
            
            
            
            
            
            plt.figure(figsize=(X_length,Y_length))
            rc('font',weight='normal',size=25) 
            clevs = np.arange(0,1.5,0.01)
            plt.contourf(x_array[:],y_array[:],U_Star[0,:,:].transpose(),cmap='viridis',levels=clevs,extend='both')#vmin=0,vmax=1)
            plt.title("u*")
            plt.colorbar()
            plt.xlim(x_array[0],x_array[-1])
            plt.ylim(y_array[0],y_array[-1])
            plt.close()
            
           
            
            plt.figure(figsize=(X_length,Y_length))
            rc('font',weight='normal',size=25) 
            clevs = np.arange(0,1.5,0.01)
            plt.contourf(x_array[:],y_array[:],U_sol[0,t_int,:,:].transpose(),cmap='viridis',levels=clevs,extend='both')#vmin=0,vmax=1)
            plt.title("u")
            plt.colorbar()
            plt.xlim(x_array[0],x_array[-1])
            plt.ylim(y_array[0],y_array[-1])
            plt.show()
            
            plt.figure(figsize=(X_length,Y_length))
            rc('font',weight='normal',size=25) 
            clevs = np.arange(0,1.5,0.01)
            plt.contourf(x_array[:],y_array[:],Con_U_sol[0,t_int,:,:].transpose(),cmap='viridis',levels=clevs,extend='both')#vmin=0,vmax=1)
            plt.title("Face u")
            plt.colorbar()
            plt.xlim(x_array[0],x_array[-1])
            plt.ylim(y_array[0],y_array[-1])
            plt.close()
            
            plt.figure(figsize=(X_length,Y_length))
            rc('font',weight='normal',size=25) 
            clevs = np.arange(0,1.5,0.01)
            plt.contourf(x_array[:],y_array[:],U_Star[1,:,:].transpose(),cmap='viridis',levels=clevs,extend='both')#vmin=0,vmax=1)
            plt.title("v*")
            plt.colorbar()
            plt.xlim(x_array[0],x_array[-1])
            plt.ylim(y_array[0],y_array[-1])
            plt.close()
            
            plt.figure(figsize=(X_length,Y_length))
            rc('font',weight='normal',size=25) 
            clevs = np.arange(-1.5,1.5,0.01)
            plt.contourf(x_array[:],y_array[:],U_sol[1,t_int,:,:].transpose(),cmap='viridis',levels=clevs,extend='both')#vmin=0,vmax=1)
            plt.title("v")
            plt.colorbar()
            plt.xlim(x_array[0],x_array[-1])
            plt.ylim(y_array[0],y_array[-1])
            plt.close()
            
            plt.figure(figsize=(X_length,Y_length))
            rc('font',weight='normal',size=25) 
            clevs = np.arange(-1.5,1.5,0.01)
            plt.contourf(x_array[:],y_array[:],Con_U_sol[1,t_int,:,:].transpose(),cmap='viridis',levels=clevs,extend='both')#vmin=0,vmax=1)
            plt.title("Face v")
            plt.colorbar()
            plt.xlim(x_array[0],x_array[-1])
            plt.ylim(y_array[0],y_array[-1])
            plt.close()
            #%%
    ## Fill arrays outside the primary dx loop once convergence is met  
    U_array_full[s_int,:Nx,:Ny] = U_sol[0,Break_index,:,:]
    V_array_full[s_int,:Nx,:Ny] = U_sol[1,Break_index,:,:]
    UF_array_full[s_int,:Nx,:Ny] = Con_U_sol[0,Break_index,:,:]
    VF_array_full[s_int,:Nx,:Ny] = Con_U_sol[1,Break_index,:,:]
    Pressure_full[s_int,:Nx,:Ny] = P_sol[Break_index,:,:]
    
    Index[s_int] = Nx
    
    print("======= dx: %1.3f done ======="%(dS))
    #%%
    ## Visualize flow field to ensure soltions realistic as the code runs
# =============================================================================
#     plt.figure(figsize=(26,11))
#     rc('font',weight='normal',size=30)
#     plt.streamplot(x_array[:],y_array[:],U_array_full[s_int,:Nx,:Ny].transpose(),V_array_full[s_int,:Nx,:Ny].transpose(),color='k',arrowstyle='->',maxlength=0.5,minlength=.005,broken_streamlines=True,density=5.5)#,color='k',arrowstyle='->',maxlength=1.5,minlength=.05,broken_streamlines=False,density=1.)#,cmap='coolwarm')
#     plt.title("u")
#     plt.xlim(x_array[0],x_array[-1])
#     plt.ylim(y_array[0],y_array[-1])
#     #plt.ylim(0,1)
#     plt.show()
# =============================================================================

    plt.plot( U_sol[0,Break_index,-2,:],y_array)
    plt.ylim(y_array[0],y_array[-1])
    plt.xlim(-1,2)
    plt.show()
    plt.figure(figsize=(26,11))
    rc('font',weight='normal',size=30)
    clevs = np.arange(0,1.5,0.01)
    plt.contourf(x_array[:],y_array[:],U_sol[0,Break_index,:,:].transpose(),cmap='coolwarm',levels=clevs)#vmin=0,vmax=1)
    plt.xlim(x_array[0],x_array[-1])
    plt.ylim(y_array[0],y_array[-1])
    plt.colorbar()
    plt.show() 
    plt.figure(figsize=(X_length,Y_length))
    rc('font',weight='normal',size=30)
    clevs = np.arange(-.5,0.5,0.1)
  

#%%
# ============================================================================
# Save cell center velocities, cell face velocites, and pressure values for 
# graphing and analysis

if save == True:
    print("Saving....")
    dataset = nc.Dataset('/Users/isaac/desktop/CFD4_RE1000_V2.nc', 
                         'w', clobber=True, format='NETCDF3_64BIT')
    dataset.title = 'Navier Stokes Solver. - Driven Cavity - CFD HW 4'
    dataset.author = 'Isaac Medina'
    dataset.contact = 'imedina2@jhu.edu'
    
    # Define dimensions
    dataset.createDimension('grids', len(Ds_array)) 
    dataset.createDimension('x', len(max_x_array))
    dataset.createDimension('y', len(max_x_array))
    
    
    # Define Variables
    dataset.createVariable('grid_sizes', 'f8', ('grids',))[:] = Ds_array
    setattr(dataset.variables['grid_sizes'],'units','Dx - dimensionless ')
    setattr(dataset.variables['grid_sizes'],'description','Grid Sizes for Analysis: 1/16,1/32,1/64,1/128')
    
    dataset.createVariable('x', 'f4', ('grids','x',))[:] = max_x_array
    setattr(dataset.variables['x'],'units','X - dimensionless ')
    setattr(dataset.variables['x'],'description','X grid per Grid Size - fill = 0s')
     
    # here x = y, change if not on a square regularly spaced grid
    dataset.createVariable('y', 'f4', ('grids','y',))[:] = max_x_array 
    setattr(dataset.variables['y'],'units','Y - dimensionless ')
    setattr(dataset.variables['y'],'description','Y grid per Grid Size - fill = 0s')
    
    dataset.createVariable('i', 'f4', ('grids',))[:] = Index
    setattr(dataset.variables['i'],'units','Indexes per grid size, X = 0, Y=1')
    setattr(dataset.variables['i'],'description','Indexes to subset data, X = 0, Y=1')
    
    dataset.createVariable('U', 'f4', ('grids','x','y',))[:] = U_array_full
    setattr(dataset.variables['U'],'units','U dimensionless')
    setattr(dataset.variables['U'],'description','Cell Center U Velocity per grid size')
    
    dataset.createVariable('V', 'f4', ('grids','x','y',))[:] = V_array_full
    setattr(dataset.variables['V'],'units','V dimensionless')
    setattr(dataset.variables['V'],'description','Cell Center V Velocity per grid size')
    
    dataset.createVariable('UF', 'f4', ('grids','x','y',))[:] = UF_array_full
    setattr(dataset.variables['UF'],'units','U dimensionless')
    setattr(dataset.variables['UF'],'description','Cell Face U Velocity per grid size')
    
    dataset.createVariable('VF', 'f4', ('grids','x','y',))[:] = VF_array_full
    setattr(dataset.variables['VF'],'units','V dimensionless')
    setattr(dataset.variables['VF'],'description','Cell Face V Velocity per grid size')
    
    dataset.createVariable('P', 'f4', ('grids','x','y',))[:] = Pressure_full
    setattr(dataset.variables['P'],'units','P dimensionless')
    setattr(dataset.variables['P'],'description','Pressure values per grid size')
    
    
    print("file_saved")
    
    # Close the file
    dataset.close()




  
# -*- coding: utf-8 -*-

import easygui
import plxscripting.easy
from plxscripting.easy import *

import numpy as np

#==============================================================================
# Path of your file
#==============================================================================

path = 'Insert the path to the file'

#==============================================================================
# Temporal Monitoring points (days)
#==============================================================================

temporal_monitor = np.array([62.165,   66.495,  72.990,  81.651,  88.146, 103.302,
                             116.292, 133.613, 150.934, 185.576, 220.218, 263.520,
                             315.483, 367.445, 402.087, 425.000])

embankment_construction = 60.0 # days

time_tt = temporal_monitor - embankment_construction

#==============================================================================
# The coordinates of Spatial monitoring points p1, p2 p3 and p4
#==============================================================================

uy_coords_data = [(40,-0.5),(40,-1.0),(40,-2.0),(40,-3.0)]

#======================================================================================================
# Noisy synthetic monitoring data of vertical displacement for the considered spatial monitoring points 
#======================================================================================================

p1 = [-3.51E-01,
-3.71E-01,
-4.02E-01,
-4.42E-01,
-4.67E-01,
-5.22E-01,
-5.62E-01,
-6.11E-01,
-6.47E-01,
-7.16E-01,
-7.70E-01,
-8.23E-01,
-8.67E-01,
-8.94E-01,
-9.15E-01,
-9.27E-01
]

p2 = [-2.88E-01,
-3.04E-01,
-3.30E-01,
-3.62E-01,
-3.84E-01,
-4.32E-01,
-4.67E-01,
-5.10E-01,
-5.43E-01,
-6.07E-01,
-6.58E-01,
-7.08E-01,
-7.49E-01,
-7.75E-01,
-7.96E-01,
-8.07E-01
]

p3 = [-2.18E-01,
-2.29E-01,
-2.47E-01,
-2.69E-01,
-2.84E-01,
-3.18E-01,
-3.44E-01,
-3.77E-01,
-4.03E-01,
-4.54E-01,
-4.96E-01,
-5.38E-01,
-5.74E-01,
-5.97E-01,
-6.15E-01,
-6.25E-01
]

p4 = [-1.71E-01,
-1.79E-01,
-1.91E-01,
-2.08E-01,
-2.19E-01,
-2.44E-01,
-2.63E-01,
-2.88E-01,
-3.08E-01,
-3.48E-01,
-3.82E-01,
-4.16E-01,
-4.47E-01,
-4.66E-01,
-4.81E-01,
-4.90E-01
]

uy_monitor_data = np.array([p1,p2,p3,p4])

state_pop             =  uy_monitor_data.shape[0]

monitoring_time_nrs   =  uy_monitor_data.shape[1]

#==============================================================================
# Initialisation
#==============================================================================

particles_prior = np.load(path + 'particles_prior.npy') # Import the ensembles

nparams = particles_prior.shape[0] # Number of parameters

N = particles_prior.shape[1] # Total number of ensembles

particles_ssm = np.zeros((nparams, N, monitoring_time_nrs+1)) # Initialise the particle space for DA

particles_ssm[:,:,0] =  particles_prior # Define the initial prior

uy_tot = np.zeros((N, state_pop)) # Initialise the state space for DA 

joint_state_param_pop = state_pop + nparams # Dimension of augmented system state

H = np.zeros((state_pop, joint_state_param_pop)) # Observation operator

H[:state_pop,:state_pop] = np.eye(state_pop) 

measurement_error = 5.0e-3 # Measurement error in m

R = np.eye(state_pop) * (measurement_error)**2 # Observation error covariance (simplified)

#==============================================================================
# Boiler plate
#==============================================================================

localhostport_input = 10000

s_i, g_i = new_server('localhost', localhostport_input)

s_i.new()

g_i.SoilContour.initializerectangular(0, -6, 80, 4) 

g_i.setproperties("Title", "emb", "Company", "Ramboll Group A/S", "Comments", "",
                  "UnitForce", "kN", "UnitLength", "m", "UnitTime", "day",
                  "UnitTemperature", "K", "UnitEnergy", "kJ", "UnitPower",
                  "kW", "WaterWeight", 10, "ReferenceTemperature", 293.15,
                  "LiquidSpecificHeatCapacity", 4181.3, "LiquidThermalConductivity", 0.0006,
                  "LiquidLatentHeat", 334000, "LiquidThermalExpansion", 0.00021, "LiquidTemperature", 293.15,
                  "IceSpecificHeatCapacity", 2108, "IceThermalConductivity" ,0.00222, "IceThermalExpansion" ,5E-5,
                  "VapourSpecificHeatCapacity", 1930 ,"VapourThermalConductivity", 2.5E-5,
                  "VapourSpecificGasConstant", 461.5, "UseTemperatureDepWaterPropsTable",
                  False, "ModelType", "Plane strain", "ElementType", "15-Noded")

#==============================================================================
# Assign Borehole
#==============================================================================

borehole_g = g_i.borehole(0)

g_i.soillayer(0)

g_i.Soillayer_1.Zones[0].Bottom = -18

g_i.Borehole_1.Head = 0.0

#==============================================================================
# Create and assign material properties
#==============================================================================

Embankment  =  g_i.soilmat()

Embankment.setproperties("Comments" ,"", "Metadata" ,"", "MaterialName", "Embankment",
                          "Colour" ,15262369, "MaterialNumber" ,1, "SoilModel" ,2 ,
                          "UserDefinedIndex" ,0, "DrainageType" ,0, "DilatancyCutOff" ,False,
                          "UndrainedBehaviour" ,0, "InterfaceStrength" ,0, "InterfaceStiffness", 0,
                          "ConsiderGapClosure" ,True, "K0PrimaryIsK0Secondary" ,True,
                          "K0Determination" ,1, "SolidThermalExpansion" ,1, "UnfrozenWaterContent", 0,
                          "TableUnfrozenWaterContent" ,"[]", "CrossPermeability" ,0,
                          "DefaultValuesAdvanced" ,True, "DataSetFlow" ,0, "ModelFlow" ,0,
                          "UDModelFlow" ,0, "SoilTypeFlow" ,0, "LowerUpper" ,0, "UsdaSoilType" ,0,
                          "StaringUpperSoilType" ,0, "StaringLowerSoilType" ,0, "M50" ,0,
                          "UseDefaultsFlow" ,0 ,
                          "TablePsiPermSat", "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
                          "SplinePsiPerm", "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
                          "SplinePsiSat", "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
                          "TensionCutOff" ,True, "einit" ,0.499999999999999, "emin" ,0, "emax" ,998.999999999999,
                          "Dinter" ,0, "SkemptonB" ,0.969899665551839, "KwRefN" ,1432098.7654321,
                          "VolumetricSpecificStorage" ,6.98275862068967E-6 ,"SolidThermalExpansionTotal" ,0,
                          "DrainageConductivity" ,0, "Eref" ,40000, "Eoed" ,64197.5308641975, "Vs", 92.4609085389233,
                          "Vp" ,192.472729585224, "Einc" ,0, 14814.8148148148, 0.35 ,35, 1, 0 ,0 ,0 ,0 ,17 ,17 ,1 ,1 ,1 ,1 ,0 ,0,
                          0 ,0 ,0 ,0.333333333333333 ,0 ,0.999 ,0 ,0 ,0 ,0 ,1.0000000000000000E+015 ,0.495 ,0 ,0 ,0 ,0 ,0 ,0 ,0,
                          0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0,
                          0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0.426423563648954 ,0.426423563648954 ,1,
                          0 ,2 ,-1.06 ,-2.37 ,0 ,0 ,10000 ,1 ,0.0620347394540943 ,3.83 ,1.3774 ,1.25 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0,
                          0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0,
                          0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,10 ,13 ,77 ,0 ,0 ,0 ,0 ,0 ,0 ,0)

Clay  =  g_i.soilmat()

Clay.setproperties("Comments" ,"", "Metadata" ,"", "MaterialName" ,"clay", "Colour" ,10283244,
                    "MaterialNumber" ,2, "SoilModel" ,5, "UserDefinedIndex" ,0, "DrainageType" ,1,
                    "DilatancyCutOff" ,False, "UndrainedBehaviour" ,0, "InterfaceStrength" ,0 ,
                    "InterfaceStiffness" ,0, "ConsiderGapClosure" ,True, "K0PrimaryIsK0Secondary" ,True,
                    "K0Determination" ,1, "SolidThermalExpansion" ,1, "UnfrozenWaterContent" ,0 ,
                    "TableUnfrozenWaterContent" ,"[]", "CrossPermeability" ,0, "DefaultValuesAdvanced" ,False,
                    "DataSetFlow" ,0, "ModelFlow" ,0, "UDModelFlow" ,0, "SoilTypeFlow" ,0, "LowerUpper" ,0 ,
                    "UsdaSoilType" ,0, "StaringUpperSoilType" ,0, "StaringLowerSoilType" ,0, "M50" ,0 ,
                    "UseDefaultsFlow" ,0 ,
                    "TablePsiPermSat" ,"[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]" ,
                    "SplinePsiPerm" ,"[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]" ,
                    "SplinePsiSat", "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]" ,
                    "UseAlternatives" ,False, "TensionCutOff" ,True, "einit" ,2, "emin" ,0, "emax" ,998.999999999999,
                    "Dinter" ,0, "SkemptonB" ,0.986622073578595, "KwRefN" ,245833.333333333 ,
                    "VolumetricSpecificStorage" ,4.0677966101695E-5, "SolidThermalExpansionTotal", 0,
                    "DrainageConductivity" ,0, "Cc" ,1.932, "Cs" ,0.1035 ,2500 ,0.2 ,30 ,0 ,0 ,0 ,0 ,0 ,17 ,17 ,0.006 ,0.006,
                    1 ,0.006 ,0 ,0 ,0 ,0 ,0 ,0.666666666666667 ,0 ,0.999 ,0 ,0 ,0 ,0 ,0.81 ,0.495 ,0 ,0 ,1.63864014030963 ,
                    9.33333333333333 ,0 ,1 ,100 ,0 ,0.5 ,0.03 ,0.28 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0,
                    0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,
                    1.0000000000000000E+010 ,1.0000000000000000E+010 ,1 ,16 ,2 ,-1.06 ,-2.37 ,0 ,0 ,10000 ,1 ,0.0620347394540943,
                    3.83 ,1.3774 ,1.25 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0,
                    0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,10 ,13 ,77 ,0 ,0 ,0 ,0 ,0 ,0 ,0)

g_i.gotostructures()

# Embankment dimensions
g_i.polygon((32.0, 0.00), (36.00, 2.00), (44.00, 2.00), (48.00, 0.00))
g_i.polygon((36.0, 2.00), (38.00, 3.00), (42.00, 3.00), (44.00, 2.00))

# Assign the spatial monitoring points
for j in range(4):

    g_i.point(uy_coords_data[j][0],uy_coords_data[j][1])

# Assign the soil material properties
g_i.Soil_1.setmaterial(Clay)
g_i.Soil_2.setmaterial(Embankment)
g_i.Soil_3.setmaterial(Embankment)

#==============================================================================
# Meshing
#==============================================================================

g_i.gotomesh()

g_i.mesh(0.01)

localhostport_o = g_i.selectmeshpoints()

s_o, g_o = new_server('localhost', localhostport_o)

for ss in range(4):
    
    g_o.addcurvepoint("node", uy_coords_data[ss][0], uy_coords_data[ss][1])

g_o.update()  # updates & closes Output

s_o.close()

#==============================================================================
# Staged construction
#==============================================================================

g_i.gotostages()

#-----------------------------------------------------------------------------

InitialPhase = g_i.InitialPhase

g_i.GroundwaterFlow.BoundaryXMin[InitialPhase]   =  "Open"
g_i.GroundwaterFlow.BoundaryXMax[InitialPhase]   =  "Open"
g_i.GroundwaterFlow.BoundaryYMin[InitialPhase]   =  "Open"
g_i.GroundwaterFlow.BoundaryYMax[InitialPhase]   =  "Open"

#-----------------------------------------------------------------------------

Phase_1 = g_i.phase(InitialPhase)

Phase_1.Identification                   =   "First Embankment construction"
Phase_1.DeformCalcType                   =   "Consolidation"
Phase_1.TimeInterval                     =    embankment_construction
Phase_1.Deform.UseUpdatedMesh            =    True
Phase_1.Deform.UseUpdatedWaterPressures  =    True

g_i.Polygon_1_1.activate(Phase_1)

#-----------------------------------------------------------------------------

Phase_2 = g_i.phase(Phase_1)

Phase_2.Identification                   =     "First Consolidation"
Phase_2.DeformCalcType                   =     "Consolidation"
Phase_2.TimeInterval                     =      365
Phase_2.Deform.UseUpdatedMesh            =      True
Phase_2.Deform.UseUpdatedWaterPressures  =      True

#==============================================================================
# Assimilation window
#==============================================================================

perform_da = True

import time

if perform_da:

    for iteration in range(1,monitoring_time_nrs):

        obs = uy_monitor_data[:,iteration-1]

        particles_ssm[:,:,iteration] = particles_ssm[:,:,iteration-1]

        ITER_TIME = []

        for nn in range(N):

            start_time = time.time()

            Clay.kappaModified                   =    particles_ssm [0,nn,iteration ]
            Clay.nu                              =    particles_ssm [1,nn,iteration ]
            Clay.lambdaModified                  =    particles_ssm [2,nn,iteration ]
            Clay.phi                             =    particles_ssm [3,nn,iteration ]
            Clay.POP                             =    particles_ssm [4,nn,iteration ]
            Clay.perm_primary_horizontal_axis    =    particles_ssm [5,nn,iteration ]
            Clay.perm_vertical_axis              =    particles_ssm [5,nn,iteration ]

            Phase_2.TimeInterval                 =   time_tt[iteration-1]

            g_i.calculate(True)

            port_o = g_i.view(g_i.Phase_2)

            s_o, g_o = new_server('localhost', port_o)

            phaseorder = [g_o.Phase_2]

            for nodes_uy in range(4):

                uy_tot[nn,nodes_uy] = g_o.getsingleresult( g_o.Phase_2,
                                                          g_o.ResultTypes.Soil.Uy,
                                                          (uy_coords_data[nodes_uy][0],
                                                           uy_coords_data[nodes_uy][1]))

            g_o.close()

            end_time = time.time()

            ITER_TIME.append(end_time-start_time)

            print('Temporal Monitoring ID: {}, Time: {}, Sample nr: {}, Calc_speed: {}'.format(iteration, time_tt[iteration-1]+60.0, nn+1, end_time-start_time))

        state_param = np.vstack((uy_tot.T, particles_ssm[:,:,iteration]))

        state_param_mean = np.mean(state_param, axis=1)

        P = (1/(N-1)) * (state_param - state_param_mean.reshape(-1,1)) @ (state_param - state_param_mean.reshape(-1,1)).T

        print('Start Data Assimilation')

        # Compute Kalman gain
        D = H @ P @ H.T + R

        KG = P @ H.T @ np.linalg.inv(D)

        wi = np.zeros((state_pop,N))

        # Create perturbed observations
        for p in range(N):
            wi[:,p] = obs + np.diag(np.random.normal(np.zeros(state_pop), R))

        # Calculate the posterior ensembles
        state_param_posterior = state_param + KG @ (wi - H @ state_param)

        state_param_posterior_mean = np.mean(state_param_posterior, axis=1)

        particles_ssm[:,:,iteration] = state_param_posterior[-nparams:,:].copy()

        # ==================== SAVE NECESSARY RESULTS ===============================

        np.save(path+'posterior_mean_{}_V21.npy'.format(iteration), state_param_posterior_mean )

        np.save(path+'particles_{}_V21.npy'.format(iteration), particles_ssm[:,:,iteration] )

        np.save(path+'time_taken_{}_V21'.format(iteration), np.array(ITER_TIME))

        print('Observations incorporated. End Data Assimilation !')


#==============================================================================
# Prediction window
#==============================================================================

Phase_3 = g_i.phase(Phase_2)

Phase_3.Identification                   =   "First Embankment construction"
Phase_3.DeformCalcType                   =   "Consolidation"
Phase_3.TimeInterval                     =    embankment_construction
Phase_3.Deform.UseUpdatedMesh            =    True
Phase_3.Deform.UseUpdatedWaterPressures  =    True

g_i.Polygon_2_1.activate(Phase_3)

#-----------------------------------------------------------------------------

Phase_4 = g_i.phase(Phase_3)

Phase_4.Identification                   =     "First Consolidation"
Phase_4.DeformCalcType                   =     "Consolidation"
Phase_4.TimeInterval                     =      715
Phase_4.Deform.UseUpdatedMesh            =      True
Phase_4.Deform.UseUpdatedWaterPressures  =      True

#-----------------------------------------------------------------------------

g_i.calculate(True)

print('Prediction window calculated')

# SAVE THE NECESSARY RESULTS IN THE PREDICTION WINDOW
#-----------------------------------------------------------------------------
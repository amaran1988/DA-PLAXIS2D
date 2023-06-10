# -*- coding: utf-8 -*-

"""

Created on Thu Jan  5 12:13:56 2023

@author: APAISE

"""

import easygui
import plxscripting.easy
from plxscripting.easy import *

import numpy as np


scale = 1

# path = 'C://Users//APAISE//OneDrive - Ramboll//Desktop//DA_ssm_v21_uy_ux4//'
path = 'C://Users//APAISE//Desktop//DA_ssm_v21_uy4//'
# path = "D://The final game//PLAXIS Files//01_one_layer//XX_finalized//reliability//DA_r_V21_full_width//Soft_Soil_Model//without_cohesion//DA_ssm_v21_uy_ux4//"


uy_monitor_data = np.load(path+'uy_monitor.npy')

time_tt = np.load(path + 'time.npy')

monitoring_time_nrs = len(time_tt)



xcoord_all = [
29,
29,
29,
29,
29,
29,
28,
28,
28,
28,
28,
28,
22,
22,
22,
22,
22,
22,
23,
23,
23,
23,
23,
23]


ycoord_all = [
-0.5,
-1,
-2,
-3,
-4,
-6,
-0.5,
-1,
-2,
-3,
-4,
-6,
-0.5,
-1,
-2,
-3,
-4,
-6,
-0.5,
-1,
-2,
-3,
-4,
-6]



uy_coords_data = np.load(path + 'uy_coords_data.npy')

step = 1.0

monitoring_points_uy = uy_monitor_data.shape[0]


monitoring_time_nrs = uy_monitor_data.shape[1]


state_pop = uy_monitor_data.shape[0]

print(state_pop)


particles_all = np.load(path + 'particles_all_r_50_prior_2_uniform.npy')
# particles_all = np.load(path + 'particles_all_r_50_prior_2_lognormal.npy')



particles_mcc_prior = particles_all

nparams = particles_mcc_prior.shape[0]

N = particles_mcc_prior.shape[1]

particles_mcc = np.zeros((nparams, N, monitoring_time_nrs+1))

particles_mcc[:,:,0] =  particles_mcc_prior

uy_tot = np.zeros((N, monitoring_points_uy))

joint_state_param_pop = state_pop + nparams


H = np.zeros((state_pop, joint_state_param_pop))

H[:state_pop,:state_pop] = np.eye(state_pop)



R = np.eye(state_pop) * (5.0e-3 * scale)**2


posterior_cov = []

prior_cov = []



print(nparams, N, joint_state_param_pop, state_pop, monitoring_time_nrs)



'''
===============================================================================
Boiler plate
===============================================================================
'''

localhostport_input = 10000

s_i, g_i = new_server('localhost', localhostport_input)


s_i.new()


g_i.SoilContour.initializerectangular(0, -6, 60, 4)


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



borehole_g = g_i.borehole(0)

g_i.soillayer(0)

g_i.Soillayer_1.Zones[0].Bottom = -18

g_i.Borehole_1.Head = -0.0

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


g_i.polygon((21.0, 0.00), (25.00, 2.00), (33.00, 2.00), (37.00, 0.00))
g_i.polygon((25.0, 2.00), (27.00, 3.00), (31.00, 3.00), (33.00, 2.00))


for jjj in range(24):

    g_i.point(xcoord_all[jjj],ycoord_all[jjj])


g_i.Soil_1.setmaterial(Clay)
g_i.Soil_2.setmaterial(Embankment)
g_i.Soil_3.setmaterial(Embankment)



'''
===============================================================================
Meshing
===============================================================================
'''

g_i.gotomesh()

g_i.mesh(0.06)


localhostport_o = g_i.selectmeshpoints()

s_o, g_o = new_server('localhost', localhostport_o)

g_o.addcurvepoint("node", 21.00, 0.00)


for ss in range(24):
    
    g_o.addcurvepoint("node", xcoord_all[ss], ycoord_all[ss])


g_o.update()  # updates & closes Output


s_o.close()



'''
===============================================================================
Staged construction
===============================================================================
'''

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
Phase_1.TimeInterval                     =    60
Phase_1.Deform.UseUpdatedMesh            =    True
Phase_1.Deform.UseUpdatedWaterPressures  =    True

g_i.Polygon_1_1.activate(Phase_1)

#-----------------------------------------------------------------------------

phase1_safety                                  =  g_i.phase(Phase_1)
phase1_safety.DeformCalcType                   = "Safety"
phase1_safety.Deform.ResetDisplacementsToZero  =  True
phase1_safety.Deform.ResetSmallStrain          =  True



g_i.delete(phase1_safety)


Phase_2 = g_i.phase(Phase_1)

Phase_2.Identification                   =     "First Consolidation"
Phase_2.DeformCalcType                   =     "Consolidation"
Phase_2.TimeInterval                     =      365
Phase_2.Deform.UseUpdatedMesh            =      True
Phase_2.Deform.UseUpdatedWaterPressures  =      True

#-----------------------------------------------------------------------------


perform_da = True

import time


if perform_da:

    print('DA to the rescue !!!!')

    for iteration in range(1,monitoring_time_nrs):


        obs = uy_monitor_data[:,iteration-1] * scale

        particles_mcc[:,:,iteration] = particles_mcc[:,:,iteration-1]


        ITER_TIME = []


        for nn in range(N):


            start_time = time.time()


            Clay.kappaModified                   =    particles_mcc [ 0,  nn,  iteration ]
            Clay.nu                              =    particles_mcc [ 1,  nn,  iteration ]
            Clay.lambdaModified                  =    particles_mcc [ 2,  nn,  iteration ]
            Clay.phi                             =    particles_mcc [ 3,  nn,  iteration ]
            Clay.POP                             =    particles_mcc [ 4,  nn,  iteration ]
            Clay.perm_primary_horizontal_axis    =    particles_mcc [ 5,  nn,  iteration ]
            Clay.perm_vertical_axis              =    particles_mcc [ 5,  nn,  iteration ]



            Phase_2.TimeInterval    =   time_tt[iteration-1]


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

            print('Iteration nr: {}, Time: {}, Sample nr: {}, Calc_speed: {}'.format(iteration, time_tt[iteration-1]+60.0, nn+1, end_time-start_time))



        state_param = np.vstack((uy_tot.T*scale, particles_mcc[:,:,iteration]))



        print(state_param)

        print(state_param.shape)

        print(H.shape)

        print(R.shape)



        state_param_mean = np.mean(state_param, axis=1)

        P = (1/(N-1)) * (state_param - state_param_mean.reshape(-1,1)) @ (state_param - state_param_mean.reshape(-1,1)).T

        s = 1.0

        P = P*s

        prior_cov = P

        particles_mean_new = state_param_mean[-nparams:]



        print('Data Assimilation Starts')


        # compute Kalman gain
        D = H @ P @ H.T + R

        KG = P @ H.T @ np.linalg.inv(D)

        wi = np.zeros((state_pop,N))



        for p in range(N):

            wi[:,p] = obs + np.diag(np.random.normal(np.zeros(state_pop), R))



        state_param_posterior = state_param + KG @ (wi - H @ state_param)

        state_param_posterior_mean = np.mean(state_param_posterior, axis=1)

        posterior_cov = (1/(N-1)) * (state_param_posterior - state_param_posterior_mean.reshape(-1,1)) @ (state_param_posterior - state_param_posterior_mean.reshape(-1,1)).T

        particles_mean_new = state_param_posterior_mean[-nparams:]

        particles_mcc[:,:,iteration] = state_param_posterior[-nparams:,:].copy()


        np.save(path+'rposterior_cov_{}_V21.npy'.format(iteration), posterior_cov )

        np.save(path+'rposterior_mean_{}_V21.npy'.format(iteration), state_param_posterior_mean )

        np.save(path+'rparticles_{}_V21.npy'.format(iteration), particles_mcc[:,:,iteration] )

        np.save(path+'rparticles_mean_{}_V21.npy'.format(iteration), particles_mean_new )

        np.save(path+'rprior_cov_{}_V21.npy'.format(iteration), prior_cov )

        np.save(path+'rprior_mean_{}_V21.npy'.format(iteration), state_param_mean )

        np.save(path+'rtime_taken_{}_ssm_V21'.format(iteration), np.array(ITER_TIME))



        print('Data Assimilation ends !')


    final_particles = particles_mcc [ :,  :,  iteration ]

    print(particles_mcc [ :,  :,  0 ])
    print('\n')
    print(final_particles)


else:

    print('No Data assimilation')

    final_particles = particles_mcc[ :, :, 0 ]

    print(final_particles)



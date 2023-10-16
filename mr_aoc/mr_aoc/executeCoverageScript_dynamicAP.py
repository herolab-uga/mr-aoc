import rps.robotarium as Robotarium_
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from utilityFunctions_AOC import *
def executeCoverageScript_dynamicAP( N=6, resolution =0.1,coverageIterations=250, noise_level=0.1, densityFlag = True):
    AP_local_positions = np.zeros((2,N))
    densityFlag = True
    distanceToCentroidThreshold = -1
    circle = ""
    robotRadius = 0.1
    # Setup for robot 4 and robot 6
    if(N==4):
        initial_conditions = np.array(np.mat('0.4 0.2 0.0 0.8 0.25;0.1 0.7 0.1 0.5 0.25; 0 0 0 0 0'))
        partitionTransparency = 0.1
        partitionMarkerSize = 2
        x_min, x_max = 0,1 
        y_min, y_max= 0,1
        sigmaVal = 0.4
        u1 = np.array([0.25,0.25])
        envRadius = 0.75
        AP_global_position =  np.array([0.25,0.25])
    
    initial_positions = copy.deepcopy(initial_conditions)
    robotarium = Robotarium_.Robotarium(number_of_robots=N+1, show_figure=True, initial_conditions=initial_conditions,sim_in_real_time=True)
    # save density positions in local FoR and change robots' facecolor
    localDensityPose_u1 = np.zeros((2,N))
    for i in range(N):
        #robotarium.chassis_patches[i].set_facecolor(ROBOT_COLOR[i])
        localDensityPose_u1[:,i] = global_to_local(np.transpose(u1),initial_positions[:,i])
    robotarium.get_poses()
    robotarium.step()
    # Create unicycle position controller
    unicycle_position_controller = create_clf_unicycle_position_controller()
    uni_barrier_cert = create_unicycle_barrier_certificate()
    ax_robotariumFig = robotarium.axes
    boundary_points = [[x_min,y_min],[x_min,y_max],[x_max,y_max],[x_max,y_min]]  # Add the first point again to close the square
    bound_x, bound_y = zip(*boundary_points)  # Unzip the coordinates
    # Plot the boundary of the square
    square, =ax_robotariumFig.plot(bound_x, bound_y, linestyle='-',linewidth = 15,color="black")
    
    # Array and variable for storing data
    centroidDistArray = np.zeros((coverageIterations,N))
    boundaryArray = np.zeros((4,N+1))
    globalDensityArray = []
    localDensityArray = [[] for _ in range(N)]
    global_densityHandle=""
    #########################################################  Coverage Script
    for iteration in range(coverageIterations):
        current_robotspositions_unicycle = robotarium.get_poses()
        center = (current_robotspositions_unicycle[0,N],current_robotspositions_unicycle[1,N])  
        theta = np.linspace(0, 2 * np.pi, 100)  # Generate angles from 0 to 2*pi
        x_boundary = center[0] + robotRadius * np.cos(theta)
        y_boundary = center[1] + robotRadius * np.sin(theta)
        # Plot the boundary points
        if(iteration>0):
            circle.remove()
        circle, = ax_robotariumFig.plot(x_boundary, y_boundary,linestyle="-",color="red",linewidth=8)
        # modify the boudnary around AP in each iteration
        AP_global_position = np.array([current_robotspositions_unicycle[0,N],current_robotspositions_unicycle[1,N]])
        u1 = np.array([current_robotspositions_unicycle[0,N],current_robotspositions_unicycle[1,N]])
        x_min = AP_global_position[0]-envRadius
        x_max = AP_global_position[0]+envRadius
        y_min = AP_global_position[1]-envRadius
        y_max = AP_global_position[1]+envRadius
        square.remove()
        boundary_points = [[x_min,y_min],[x_min,y_max],[x_max,y_max],[x_max,y_min]]  # Add the first point again to close the square
        bound_x, bound_y = zip(*boundary_points)  # Unzip the coordinates
        square, =ax_robotariumFig.plot(bound_x, bound_y, linestyle='-',linewidth = 8,color="black")
        goalForCentroid = np.zeros((2,N+1))
        distToCentroid = np.ones((N))*5
        local_positions = np.zeros((2,N))
        # Convert the robot's positions in its own FoR
        for i in range(N):
            local_position = global_to_local(current_robotspositions_unicycle[:2, i], initial_positions[:2, i])
            local_positions[0][i]=local_position[0]
            local_positions[1][i]=local_position[1]
        for robot in range(N):
            noise = np.array([np.random.normal(0, noise_level), np.random.normal(0, noise_level)  ]) 
            AP_local_positions[:,robot] = global_to_local(AP_global_position,initial_positions[:,robot]) + noise
        # get relative position using AP location
        # get AP Pos in local FoR
        for robot in range(N):
            noise = np.array([np.random.normal(0, noise_level), np.random.normal(0, noise_level)  ]) 
            AP_local_positions[:,robot] = global_to_local(AP_global_position,initial_positions[:,robot]) + noise
        # Here range used is N+2 because N represents pursuing robots, N+1 represnts global FoR,  N+2 for AP
        for robot in range(N+2):
            # if robot in range N find other robots' positions using Vector Transformation
            if(robot<N):
                current_robotspositions_ulocal = np.zeros((2,N))
                current_robotspositions_ulocal[:2,robot] = local_positions[:2,robot]
                for neighbor in range(N):
                    if robot != neighbor:
                        ap_diff=[0,0]
                        ap_diff[0]=AP_local_positions[0,robot]-AP_local_positions[0,neighbor]
                        ap_diff[1]=AP_local_positions[1,robot]-AP_local_positions[1,neighbor]
                        current_robotspositions_ulocal[0,neighbor]= local_positions[0,neighbor] + ap_diff[0]
                        current_robotspositions_ulocal[1,neighbor]=local_positions[1,neighbor] + ap_diff[1]
            # Get region boundary
            if(iteration<coverageIterations):
                if(robot<N):
                    min_x = AP_local_positions[0,robot]-envRadius
                    max_x = AP_local_positions[0,robot]+envRadius
                    min_y = AP_local_positions[1,robot]-envRadius
                    max_y = AP_local_positions[1,robot]+envRadius
                    boundaryArray[:,robot] = np.array([min_x,max_x,min_y,max_y])
                if robot==N:
                    boundaryArray[:,robot] = np.array([x_min,x_max,y_min,y_max]) 
            if(robot<N):  
                if densityFlag:
                    localDensityArray[robot],densityHandle = getDensityArray_robotarium(ax_robotariumFig,boundaryArray[0,robot],boundaryArray[1,robot],boundaryArray[2,robot],boundaryArray[3,robot],resolution,np.transpose(localDensityPose_u1[:,robot]),sigmaVal,False)
                current_local_pose = np.transpose(current_robotspositions_ulocal)
                C_x, C_y , cost, area,local_hull_figHandles, local_Hull_textHandles = partitionFinder_robotariumm(ax_robotariumFig,(current_local_pose), [boundaryArray[0,robot],boundaryArray[1,robot]], [boundaryArray[2,robot],boundaryArray[3,robot]], resolution, densityFlag ,localDensityArray[robot], partitionTransparency,partitionMarkerSize, False)                  
                centroid = np.transpose(np.array([C_x,C_y]))
                robotsPositions_diff = np.round (centroid[robot,:] - current_local_pose[robot,:2],2)
                distToCentroid[robot] = (math.sqrt((current_local_pose[robot, 0] - C_x[robot]) ** 2 + (current_local_pose[robot, 1] - C_y[robot]) ** 2))
                distToCentroid[robot] =  round(distToCentroid[robot], 2)
                centroidDistArray[iteration,robot] = distToCentroid[robot]
                local_goal_location = current_robotspositions_ulocal[:,robot] + robotsPositions_diff
                # convert locally obtained goal to global goal for robotarium
                goalForCentroid[:,robot] = local_to_global(local_goal_location,initial_positions[:2,robot])
            # if robot==N,golbal FoR is handled here (all local positions are converted to global to set robotarium velocity)
            if(robot==N):   
                if(densityFlag):
                        if(iteration>0):
                            for robot_r in range(N):
                                global_hull_textHandles[robot_r].remove()
                                hullObject = global_hull_figHandles[robot_r]
                                hullObject.remove()
                            global_densityHandle.remove()
                        globalDensityArray, global_densityHandle = getDensityArray_robotarium(ax_robotariumFig,x_min,x_max,y_min,y_max,resolution,u1,sigmaVal,True)
                C_x, C_y , cost, global_area, global_hull_figHandles, global_hull_textHandles = partitionFinder_robotariumm(ax_robotariumFig,np.transpose(current_robotspositions_unicycle[:2,:N]), [x_min,x_max], [y_min,y_max],  resolution, densityFlag ,globalDensityArray, partitionTransparency,partitionMarkerSize,True) 
            # handle AP position
            if(robot==N+1): 
                goalForCentroid[:,N] = np.array([current_robotspositions_unicycle[0,robot-1],current_robotspositions_unicycle[1,robot-1]])
                for robot_r in range(N):
                    localDensityPose_u1[:,robot_r] = global_to_local(np.transpose(u1),initial_positions[:,robot_r])

        # if centroidDistance value for all robot reaches threshold stop the program
        if(all(val < distanceToCentroidThreshold for val in distToCentroid)):
            robotarium.step()
            break        
        # assign goal and calculate velocities for robotarium
        dxu = unicycle_position_controller(current_robotspositions_unicycle, goalForCentroid[:2])
        if iteration<30:
            dxu[:,N] = np.array([0,0])
        else:
            dxu[:,N] = np.array([-0.05,0.05])
        if iteration>70:
            dxu[:,N] = np.array([0,0])
        dxu = uni_barrier_cert(dxu, current_robotspositions_unicycle)
        robotarium.set_velocities(np.arange(N+1), dxu)
        robotarium.step()             
    
    return 

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

def executeCoverageScript_function( N=6, resolution=0.1, coverageIterations=250, noise_level=0.1, densityFlag = True, targetTrackingFlag=True, showLocalFOR=False, envRadius_diff = []):
    AP_local_positions = np.zeros((2,N))
    if targetTrackingFlag:
        distanceToCentroidThreshold = -0.1
        densityFlag = True
    else:
        distanceToCentroidThreshold = 0.05
    robotRadius = 0.1
    ROBOT_COLOR = {0: "red", 1: "green", 2: "grey", 3:"blue",4:"orange",5:"black"}
    colorList =  [ROBOT_COLOR[key] for key in range(N)]
    # Setup for robot 4 and robot 6
    if(N==4):
        if targetTrackingFlag:
            initial_conditions = np.array(np.mat(' -0.9 0.9 0.5 -0.5 0.85; 0.0 0.0 0.0 0.0 0.85; 0 0 0 0 0'))
        else:
            initial_conditions = np.array(np.mat(' -0.9 0.9 0.5 -0.5; 0.0 0.0 0.0 0.0; 0 3.14 0.785 1.57'))
            #initial_conditions = np.array(np.mat(' -0.9 0.9 0.5 -0.5; 0.0 0.0 0.0 0.0; 0 0.785 0 0'))
        partitionTransparency = 0.1
        partitionMarkerSize = 6
        x_min, x_max = -1, 1
        y_min, y_max = -1, 1
        sigmaVal = 0.4
        u1 = np.array([0.85, 0.85])
        envRadius = 1
        
        AP_global_position =  np.array([0,0])
    # initialize envRadius_updated
    envRadius_updated = envRadius_robots
    # make an independent copy of initial robots positions to calculate local FoR locations
    initial_positions = copy.deepcopy(initial_conditions)
    if targetTrackingFlag:
        robotarium = Robotarium_.Robotarium(number_of_robots=N+1, show_figure=True, initial_conditions=initial_conditions,sim_in_real_time=True)
    else:
        robotarium = Robotarium_.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions,sim_in_real_time=True)
    
    # save density positions in local FoR and change robots' facecolor
    localDensityPose_u1 = np.zeros((2,N))
    for i in range(N):
        # comment line 53 for real-robot experiments
        robotarium.chassis_patches[i].set_facecolor(ROBOT_COLOR[i])
        localDensityPose_u1[:,i] = global_to_local(np.transpose(u1),initial_positions[:,i])
    robotarium.get_poses()
    robotarium.step()
    # Create unicycle position controller
    unicycle_position_controller = create_clf_unicycle_position_controller()
    uni_barrier_cert = create_unicycle_barrier_certificate()
    ax_robotariumFig = robotarium.axes
    # Plot workspace boundary 
    boundary_points = [[x_min,y_min],[x_min,y_max],[x_max,y_max],[x_max,y_min],[x_min,y_min]] 
    bound_x, bound_y = zip(*boundary_points) 
    square, =ax_robotariumFig.plot(bound_x, bound_y, linestyle='-',linewidth = 5,color="black")
    # Array and variable for storing data
    boundaryArray = np.zeros((4,N+1))
    globalDensityArray = []
    localDensityArray = [[] for _ in range(N)]
    global_densityHandle=""
    

    if showLocalFOR:
        ax_local_coverage = []
        fig_local_coverage = []
        for i in range(N): 
            fig_tmp =  plt.figure()
            ax_local_coverage.append(fig_tmp.add_subplot())
            #ax_local_coverage[i].set_title(f"coverage: Robot{str(i+1)}'s FoR",fontsize=20)
            ax_local_coverage[i].set_xlabel('x')
            ax_local_coverage[i].set_ylabel('y')
            fig_local_coverage.append(fig_tmp)
            fig_tmp = ""
        local_positions = [[ax_local_coverage[i].plot([], [], 'o', markersize=2, color="black", label=f'Robot {i+1}')[0] for i in range(N)]]
    #########################################################  Coverage Script
    for iteration in range(coverageIterations):
        current_robotspositions_unicycle = robotarium.get_poses()
        if targetTrackingFlag:
            center = (current_robotspositions_unicycle[0,N],current_robotspositions_unicycle[1,N])  
            theta = np.linspace(0, 2 * np.pi, 100)  # Generate angles from 0 to 2*pi
            x_boundary = center[0] + robotRadius * np.cos(theta)
            y_boundary = center[1] + robotRadius * np.sin(theta)
            if iteration>0:
                circle.remove()
            circle, = ax_robotariumFig.plot(x_boundary, y_boundary,linestyle="-",color="black",linewidth=4)
        if targetTrackingFlag:
            goalForCentroid = np.zeros((2,N+1))
        else:
            goalForCentroid = np.zeros((2,N))
        distToCentroid = np.ones((N))*5
        local_positions = np.zeros((2,N))
        # Convert the robot's positions in its own FoR
        for i in range(N):
            local_position = global_to_local(current_robotspositions_unicycle[:2, i], initial_positions[:, i])
            local_positions[0][i]=local_position[0]
            local_positions[1][i]=local_position[1]
        # get relative position using AP location
        # get AP Pos in local FoR
        for robot in range(N):
            noise = np.array([np.random.normal(0, noise_level), np.random.normal(0, noise_level)  ]) 
            AP_local_positions[:,robot] = global_to_local(AP_global_position,initial_positions[:,robot]) + noise
        # Here range used is N+1 because N represents local robot's FoR  and last index N+1 is assigned to global
        for robot in range(N+2):
            # if robot in range N find other robots' positions using Vector Transformation
            if(robot<N):
                
                current_robotspositions_ulocal = np.zeros((2,N))
                current_robotspositions_ulocal[:2,robot] = local_positions[:2,robot]
                for neighbor in range(N):
                    if robot != neighbor:
                        # passing orientation of robot and its neighbor
                        rotation_matrix = get_rotation_matrix_z(initial_positions[2,robot]-initial_positions[2,neighbor])
                        current_robotspositions_ulocal[:2,neighbor] = AP_local_positions[:2,robot]+ np.dot(rotation_matrix,(local_positions[:2,neighbor]-AP_local_positions[:2,neighbor]))
            # Get unified and local region boundaries only for the 1st iteration
            envRadius_robots = envRadius_updated
            if(iteration<coverageIterations):
                if(robot<N):
                    min_x = AP_local_positions[0,robot]-envRadius_robots[robot]
                    max_x = AP_local_positions[0,robot]+envRadius_robots[robot]
                    min_y = AP_local_positions[1,robot]-envRadius_robots[robot]
                    max_y = AP_local_positions[1,robot]+envRadius_robots[robot]
                    boundaryArray[:,robot] = np.array([min_x,max_x,min_y,max_y])
                if robot==N:
                    boundaryArray[:,robot] = np.array([x_min,x_max,y_min,y_max]) 
            # if robot is in range "N" and showLocalPerspective is True, show robots' partition in local FoR
            if(robot<N):  
                if showLocalFOR:
                    ax_robot_FoR = ax_local_coverage[robot]
                    ax_robot_FoR.cla()
                    ax_robot_FoR.set_title(f"robot{robot+1}_FoR")
                else:
                    ax_robot_FoR = None
                neighbor_range_differences = []
                # performing consenus considering all the nerighbor robot's radius
                for neighbor in range(N):
                    neighbor_range_differences.append(envRadius_robots[neighbor]-envRadius_robots[robot])
                envRadius_updated[robot] = envRadius_robots[robot] + np.mean(neighbor_range_differences)
                if densityFlag:
                    localDensityArray[robot],densityHandle = (getDensityArray_robotarium(ax_robot_FoR,boundaryArray[0,robot],boundaryArray[1,robot],boundaryArray[2,robot],
                    			boundaryArray[3,robot],resolution, np.transpose(localDensityPose_u1[:,robot]),sigmaValue=sigmaVal,displayFlag=showLocalFOR))
                current_local_pose = np.transpose(current_robotspositions_ulocal)      
                C_x, C_y , cost, area,local_hull_figHandles, local_Hull_textHandles = partitionFinder_robotariumm(ax_robot_FoR,(current_local_pose),[boundaryArray[0,robot],
                					boundaryArray[1,robot]], [boundaryArray[2,robot],boundaryArray[3,robot]], resolution, densityFlag ,localDensityArray[robot], 
                					alpha=partitionTransparency, partitionMarkerSize=partitionMarkerSize, globalFrame=False, robot_color = ROBOT_COLOR)                  
                centroid = np.transpose(np.array([C_x,C_y]))
                robotsPositions_diff = centroid[robot,:] - current_local_pose[robot,:2]
                robotsPositions_diff =  np.round(robotsPositions_diff,decimals=2)
                distToCentroid[robot] = (math.sqrt((current_local_pose[robot, 0] - C_x[robot]) ** 2 + (current_local_pose[robot, 1] - C_y[robot]) ** 2))
                distToCentroid[robot] =  round(distToCentroid[robot], 2)
                local_goal_location = current_robotspositions_ulocal[:,robot] + robotsPositions_diff
                if showLocalFOR:
                    ax_robot_FoR.scatter(local_goal_location[0],local_goal_location[1],color=ROBOT_COLOR[robot],marker = "o",facecolor= "none",s=130,linewidth=3)
                goalForCentroid[:,robot] = local_to_global(local_goal_location,initial_positions[:,robot])
            # if robot==N,golbal FoR is handled herecp
            if(robot==N):
                if(iteration>0):
                    for robot_r in range(N):
                        global_hull_textHandles[robot_r].remove()
                        hullObject = global_hull_figHandles[robot_r]
                        hullObject.remove()
                    scatter.remove()
                    if targetTrackingFlag:
                        global_densityHandle.remove()
                if densityFlag:
                    globalDensityArray, global_densityHandle = getDensityArray_robotarium(ax_robotariumFig,x_min,x_max,y_min,y_max,resolution,u1,sigmaVal,displayFlag=True)
                C_x, C_y , cost, global_area, global_hull_figHandles, global_hull_textHandles = partitionFinder_robotariumm(ax_robotariumFig,np.transpose(current_robotspositions_unicycle[:2,:N]), [x_min,x_max], [y_min,y_max], resolution, densityFlag ,globalDensityArray,alpha = partitionTransparency, partitionMarkerSize=partitionMarkerSize,globalFrame=True, robot_color = ROBOT_COLOR) 
                
                scatter = ax_robotariumFig.scatter(goalForCentroid[0,:N],goalForCentroid[1,:N],marker = "o", facecolor="none", s= 130, linewidth=3, color = colorList[:N])

            # N+1 robot here represents the dynamic AP
            if targetTrackingFlag:
                if(robot==N+1): 
                    center = (current_robotspositions_unicycle[0,robot-1],current_robotspositions_unicycle[1,robot-1])  
                    goalForCentroid[:,N] = np.array([current_robotspositions_unicycle[0,robot-1],current_robotspositions_unicycle[1,robot-1]])
                    u1 = np.array([current_robotspositions_unicycle[0,N],current_robotspositions_unicycle[1,N]])
                    for robot_r in range(N):
                        localDensityPose_u1[:,robot_r] = global_to_local(np.transpose(u1),initial_positions[:,robot_r])
                        if showLocalFOR:
                            ax_local_coverage[robot_r].scatter(localDensityPose_u1[0,robot_r],localDensityPose_u1[1,robot_r],marker = "o",facecolor= "black",s=130,linewidth=5)
        # if centroidDistance value for all robot reaches threshold stop the program
        if(all(val < distanceToCentroidThreshold for val in distToCentroid)):
            robotarium.step()
            break  
        # assign goal and calculate velocities for robotarium
        dxu = unicycle_position_controller(current_robotspositions_unicycle, goalForCentroid[:2])
        if targetTrackingFlag:
            if iteration<15 or iteration>70:
                dxu[:,N] = np.array([0,0])
            else:
                dxu[:,N] = np.array([-0.1,0.1])
        dxu = uni_barrier_cert(dxu, current_robotspositions_unicycle)
        if targetTrackingFlag:
            robotarium.set_velocities(np.arange(N+1), dxu)
        else:
            robotarium.set_velocities(np.arange(N), dxu)
        robotarium.step()             
    return 

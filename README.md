# Notice 
Imported from https://github.com/sporsho/TensorRT_OutDet 


# How to Build 
### Configure 
Open the ```CmakeList.txt```
Point the libraraies on your system. 

### Create a build folder 

``` mkdir build```
### catkin build 

```catkin_make```

### run the dummy point publisher 
1. Open a terminal 
2. cd to the catkin workspace 
3. run ```source devel/setup.bash``` 
4. run ```rosrun outdet point_publisher``` 
This will read a point cloud in KITTI format and publish it continuously. Replace the file path with your point cloud. 

### run the tensorRT inference 
1. Open a terminal
2. cd to the catkin workspace
3. run ```source devel/setup.bash```
4. run ```rosrun outdet filter_with_outdet``` 
This will run TensorRT inference in C++, the code needs to be optimized and paralellized based on the environment. The node will extract the 10 meter front view in the point cloud and run 3D-OutDet on it, then the node will publish the filtered point cloud which can be accessed with the topic name ```/desnowed_cloud``` 

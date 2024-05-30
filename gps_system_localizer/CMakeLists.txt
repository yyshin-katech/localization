cmake_minimum_required(VERSION 3.0.2)
project(gps_system_localizer)

## Compile as C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  mmc_msgs
  roslib
  tf
)

find_package(PROJ)
# SET(PROJ_INCLUDE_DIR "/usr/local/" )
# INCLUDE_DIRECTORIES ( "${PROJ_INCLUDE_DIR}" )
# MESSAGE(STATUS "${PROJ_INCLUDE_DIR}: " ${PROJ_INCLUDE_DIR})

find_package (Eigen3)
SET(EIGEN3_INCLUDE_DIR "/usr/include/eigen3/" )
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )


catkin_package(
  # INCLUDE_DIRS include
  # LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs geometry_msgs mmc_msgs roslib tf novatel_gps_msgs
  DEPENDS PROJ
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

link_directories(${catkin_LIBRARY_DIRS})

# add_library(${PROJECT_NAME} lib/gps_to_tf_lib.cpp)

# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

################################################################################ Filtering_node
add_executable(gps_to_tf src/gps_to_tf.cpp)
target_compile_options(gps_to_tf PRIVATE)
add_dependencies(gps_to_tf ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(gps_to_tf 
  # ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${PROJ_LIBRARIES}
)

# install(TARGETS ${PROJECT_NAME} gps_to_tf
install(TARGETS gps_to_tf
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
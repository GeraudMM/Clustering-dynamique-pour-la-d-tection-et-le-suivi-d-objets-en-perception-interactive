# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/geraud/Desktop/Simulation/segmentation

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/geraud/Desktop/Simulation/segmentation/build

# Include any dependencies generated for this target.
include libs/CMakeFiles/lib_papon.dir/depend.make

# Include the progress variables for this target.
include libs/CMakeFiles/lib_papon.dir/progress.make

# Include the compile flags for this target's objects.
include libs/CMakeFiles/lib_papon.dir/flags.make

libs/CMakeFiles/lib_papon.dir/papon/supervoxel/src/sequential_supervoxel_clustering.cpp.o: libs/CMakeFiles/lib_papon.dir/flags.make
libs/CMakeFiles/lib_papon.dir/papon/supervoxel/src/sequential_supervoxel_clustering.cpp.o: ../libs/papon/supervoxel/src/sequential_supervoxel_clustering.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/geraud/Desktop/Simulation/segmentation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libs/CMakeFiles/lib_papon.dir/papon/supervoxel/src/sequential_supervoxel_clustering.cpp.o"
	cd /home/geraud/Desktop/Simulation/segmentation/build/libs && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lib_papon.dir/papon/supervoxel/src/sequential_supervoxel_clustering.cpp.o -c /home/geraud/Desktop/Simulation/segmentation/libs/papon/supervoxel/src/sequential_supervoxel_clustering.cpp

libs/CMakeFiles/lib_papon.dir/papon/supervoxel/src/sequential_supervoxel_clustering.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lib_papon.dir/papon/supervoxel/src/sequential_supervoxel_clustering.cpp.i"
	cd /home/geraud/Desktop/Simulation/segmentation/build/libs && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/geraud/Desktop/Simulation/segmentation/libs/papon/supervoxel/src/sequential_supervoxel_clustering.cpp > CMakeFiles/lib_papon.dir/papon/supervoxel/src/sequential_supervoxel_clustering.cpp.i

libs/CMakeFiles/lib_papon.dir/papon/supervoxel/src/sequential_supervoxel_clustering.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lib_papon.dir/papon/supervoxel/src/sequential_supervoxel_clustering.cpp.s"
	cd /home/geraud/Desktop/Simulation/segmentation/build/libs && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/geraud/Desktop/Simulation/segmentation/libs/papon/supervoxel/src/sequential_supervoxel_clustering.cpp -o CMakeFiles/lib_papon.dir/papon/supervoxel/src/sequential_supervoxel_clustering.cpp.s

# Object files for target lib_papon
lib_papon_OBJECTS = \
"CMakeFiles/lib_papon.dir/papon/supervoxel/src/sequential_supervoxel_clustering.cpp.o"

# External object files for target lib_papon
lib_papon_EXTERNAL_OBJECTS =

libs/liblib_papon.a: libs/CMakeFiles/lib_papon.dir/papon/supervoxel/src/sequential_supervoxel_clustering.cpp.o
libs/liblib_papon.a: libs/CMakeFiles/lib_papon.dir/build.make
libs/liblib_papon.a: libs/CMakeFiles/lib_papon.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/geraud/Desktop/Simulation/segmentation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library liblib_papon.a"
	cd /home/geraud/Desktop/Simulation/segmentation/build/libs && $(CMAKE_COMMAND) -P CMakeFiles/lib_papon.dir/cmake_clean_target.cmake
	cd /home/geraud/Desktop/Simulation/segmentation/build/libs && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lib_papon.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libs/CMakeFiles/lib_papon.dir/build: libs/liblib_papon.a

.PHONY : libs/CMakeFiles/lib_papon.dir/build

libs/CMakeFiles/lib_papon.dir/clean:
	cd /home/geraud/Desktop/Simulation/segmentation/build/libs && $(CMAKE_COMMAND) -P CMakeFiles/lib_papon.dir/cmake_clean.cmake
.PHONY : libs/CMakeFiles/lib_papon.dir/clean

libs/CMakeFiles/lib_papon.dir/depend:
	cd /home/geraud/Desktop/Simulation/segmentation/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/geraud/Desktop/Simulation/segmentation /home/geraud/Desktop/Simulation/segmentation/libs /home/geraud/Desktop/Simulation/segmentation/build /home/geraud/Desktop/Simulation/segmentation/build/libs /home/geraud/Desktop/Simulation/segmentation/build/libs/CMakeFiles/lib_papon.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libs/CMakeFiles/lib_papon.dir/depend


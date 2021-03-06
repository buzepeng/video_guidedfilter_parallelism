# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/opencv_cpu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/opencv_cpu/build

# Include any dependencies generated for this target.
include CMakeFiles/gauss_gpu.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/gauss_gpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gauss_gpu.dir/flags.make

CMakeFiles/gauss_gpu.dir/src/gauss.cu.o: CMakeFiles/gauss_gpu.dir/flags.make
CMakeFiles/gauss_gpu.dir/src/gauss.cu.o: ../src/gauss.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/opencv_cpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/gauss_gpu.dir/src/gauss.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/opencv_cpu/src/gauss.cu -o CMakeFiles/gauss_gpu.dir/src/gauss.cu.o

CMakeFiles/gauss_gpu.dir/src/gauss.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/gauss_gpu.dir/src/gauss.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/gauss_gpu.dir/src/gauss.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/gauss_gpu.dir/src/gauss.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/gauss_gpu.dir/src/gauss.cu.o.requires:

.PHONY : CMakeFiles/gauss_gpu.dir/src/gauss.cu.o.requires

CMakeFiles/gauss_gpu.dir/src/gauss.cu.o.provides: CMakeFiles/gauss_gpu.dir/src/gauss.cu.o.requires
	$(MAKE) -f CMakeFiles/gauss_gpu.dir/build.make CMakeFiles/gauss_gpu.dir/src/gauss.cu.o.provides.build
.PHONY : CMakeFiles/gauss_gpu.dir/src/gauss.cu.o.provides

CMakeFiles/gauss_gpu.dir/src/gauss.cu.o.provides.build: CMakeFiles/gauss_gpu.dir/src/gauss.cu.o


CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.o: CMakeFiles/gauss_gpu.dir/flags.make
CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.o: ../src/LBBGM2.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/opencv_cpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/opencv_cpu/src/LBBGM2.cu -o CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.o

CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.o.requires:

.PHONY : CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.o.requires

CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.o.provides: CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.o.requires
	$(MAKE) -f CMakeFiles/gauss_gpu.dir/build.make CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.o.provides.build
.PHONY : CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.o.provides

CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.o.provides.build: CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.o


# Object files for target gauss_gpu
gauss_gpu_OBJECTS = \
"CMakeFiles/gauss_gpu.dir/src/gauss.cu.o" \
"CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.o"

# External object files for target gauss_gpu
gauss_gpu_EXTERNAL_OBJECTS =

CMakeFiles/gauss_gpu.dir/cmake_device_link.o: CMakeFiles/gauss_gpu.dir/src/gauss.cu.o
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.o
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: CMakeFiles/gauss_gpu.dir/build.make
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_gapi.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_stitching.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_alphamat.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_aruco.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_bgsegm.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_bioinspired.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_ccalib.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudabgsegm.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudafeatures2d.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudaobjdetect.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudastereo.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_cvv.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_dnn_objdetect.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_dnn_superres.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_dpm.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_face.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_freetype.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_fuzzy.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_hfs.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_img_hash.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_intensity_transform.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_line_descriptor.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_quality.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_rapid.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_reg.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_rgbd.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_saliency.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_stereo.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_structured_light.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_superres.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_surface_matching.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_tracking.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_videostab.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_xfeatures2d.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_xobjdetect.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_xphoto.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_shape.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_highgui.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_datasets.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_plot.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_text.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_dnn.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_ml.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_phase_unwrapping.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudacodec.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_videoio.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudaoptflow.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudalegacy.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudawarping.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_optflow.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_ximgproc.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_video.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_imgcodecs.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_objdetect.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_calib3d.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_features2d.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_flann.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_photo.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudaimgproc.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudafilters.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_imgproc.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudaarithm.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_core.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudev.so.4.3.0
CMakeFiles/gauss_gpu.dir/cmake_device_link.o: CMakeFiles/gauss_gpu.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/opencv_cpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA device code CMakeFiles/gauss_gpu.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gauss_gpu.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gauss_gpu.dir/build: CMakeFiles/gauss_gpu.dir/cmake_device_link.o

.PHONY : CMakeFiles/gauss_gpu.dir/build

# Object files for target gauss_gpu
gauss_gpu_OBJECTS = \
"CMakeFiles/gauss_gpu.dir/src/gauss.cu.o" \
"CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.o"

# External object files for target gauss_gpu
gauss_gpu_EXTERNAL_OBJECTS =

gauss_gpu: CMakeFiles/gauss_gpu.dir/src/gauss.cu.o
gauss_gpu: CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.o
gauss_gpu: CMakeFiles/gauss_gpu.dir/build.make
gauss_gpu: /usr/local/lib/libopencv_gapi.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_stitching.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_alphamat.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_aruco.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_bgsegm.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_bioinspired.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_ccalib.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_cudabgsegm.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_cudafeatures2d.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_cudaobjdetect.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_cudastereo.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_cvv.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_dnn_objdetect.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_dnn_superres.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_dpm.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_face.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_freetype.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_fuzzy.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_hfs.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_img_hash.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_intensity_transform.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_line_descriptor.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_quality.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_rapid.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_reg.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_rgbd.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_saliency.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_stereo.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_structured_light.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_superres.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_surface_matching.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_tracking.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_videostab.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_xfeatures2d.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_xobjdetect.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_xphoto.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_shape.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_highgui.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_datasets.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_plot.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_text.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_dnn.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_ml.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_phase_unwrapping.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_cudacodec.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_videoio.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_cudaoptflow.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_cudalegacy.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_cudawarping.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_optflow.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_ximgproc.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_video.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_imgcodecs.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_objdetect.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_calib3d.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_features2d.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_flann.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_photo.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_cudaimgproc.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_cudafilters.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_imgproc.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_cudaarithm.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_core.so.4.3.0
gauss_gpu: /usr/local/lib/libopencv_cudev.so.4.3.0
gauss_gpu: CMakeFiles/gauss_gpu.dir/cmake_device_link.o
gauss_gpu: CMakeFiles/gauss_gpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/opencv_cpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CUDA executable gauss_gpu"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gauss_gpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gauss_gpu.dir/build: gauss_gpu

.PHONY : CMakeFiles/gauss_gpu.dir/build

CMakeFiles/gauss_gpu.dir/requires: CMakeFiles/gauss_gpu.dir/src/gauss.cu.o.requires
CMakeFiles/gauss_gpu.dir/requires: CMakeFiles/gauss_gpu.dir/src/LBBGM2.cu.o.requires

.PHONY : CMakeFiles/gauss_gpu.dir/requires

CMakeFiles/gauss_gpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gauss_gpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gauss_gpu.dir/clean

CMakeFiles/gauss_gpu.dir/depend:
	cd /home/opencv_cpu/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/opencv_cpu /home/opencv_cpu /home/opencv_cpu/build /home/opencv_cpu/build /home/opencv_cpu/build/CMakeFiles/gauss_gpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gauss_gpu.dir/depend


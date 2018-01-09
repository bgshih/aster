cmake_minimum_required(VERSION 2.8)
project(rare)

# compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${OpenMP_CXX_FLAGS} -Wall -fPIC -D_GLIBCXX_USE_CXX11_ABI=0")

# TensorFlow dependencies
execute_process(COMMAND python3 -c "import os, sys; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; sys.stdout.write(tf.sysconfig.get_include()); sys.stdout.flush()" OUTPUT_VARIABLE TF_INC)
message(STATUS "Found TF_INC: " ${TF_INC})
execute_process(COMMAND python3 -c "import os, sys; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; sys.stdout.write(tf.sysconfig.get_lib()); sys.stdout.flush()" OUTPUT_VARIABLE TF_LIB)
message(STATUS "Found TF_LIB: " ${TF_LIB})

# target
include_directories(${TF_INC} "${TF_INC}/external/nsync/public")
link_directories(${TF_LIB})
add_library(rare SHARED 
  string_filtering_op.cc
  string_reverse_op.cc
  divide_curve_op.cc)
target_link_libraries(rare tensorflow_framework)

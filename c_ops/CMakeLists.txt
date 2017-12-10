CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(rare)

# compiler flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${OpenMP_CXX_FLAGS} -Wall -fPIC -D_GLIBCXX_USE_CXX11_ABI=0")

# TensorFlow dependencies
EXECUTE_PROCESS(COMMAND python3 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(tf.sysconfig.get_include())"
                OUTPUT_VARIABLE TF_INC)
MESSAGE(STATUS "Found TF_INC: " ${TF_INC})

# target
INCLUDE_DIRECTORIES(${TF_INC})
ADD_LIBRARY(rare SHARED 
  string_filtering_op.cc)

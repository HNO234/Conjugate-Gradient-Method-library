WORKING_DIR = $(shell pwd)
#Flags
CXX = /usr/bin/g++-10
NVCC = nvcc
FLAGS_CPU = -g -O3 -m64 -Wall -shared -std=c++17 -fPIC -fopenmp -pthread -ffast-math -ftree-vectorize
FLAGS_GPU_COMPILE = -g -O3 -m64 -shared -std=c++17 -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' --compiler-bindir=$(CXX)
FLAGS_GPU_LINK = -shared -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' --compiler-bindir=$(CXX)
PYBINCLUDE = $(shell python3-config --includes) $(shell python3 -m pybind11 --includes)
FLAGS_DEP = -MMD -MP
DIRS_CPU = $(WORKING_DIR)/cpp/matrix/naive $(WORKING_DIR)/cpp/cg_method $(WORKING_DIR)/cpp/matrix/cpu
CXXINCLUDE_CPU := $(patsubst %,-I %,$(DIRS_CPU))
DIRS_GPU = $(WORKING_DIR)/cpp/matrix/gpu
CXXINCLUDE_GPU := $(patsubst %,-I %,$(DIRS_GPU))
#PATH
MODULE_SHARE_OBJS_RLT_DIR = cpp
MODULE_SHARE_OBJS_ABS_DIR = $(WORKING_DIR)/$(MODULE_SHARE_OBJS_RLT_DIR)
PYTHONPATH := $(MODULE_SHARE_OBJS_ABS_DIR):$(PYTHONPATH)
export PYTHONPATH

#Includes
CPP_FILE_CPU = $(wildcard $(MODULE_SHARE_OBJS_RLT_DIR)/matrix/cpu/*.cpp)
CPP_FILE_GPU = $(wildcard $(MODULE_SHARE_OBJS_RLT_DIR)/matrix/gpu/*.cu)
CPP_FILE_SHARED = $(wildcard $(MODULE_SHARE_OBJS_RLT_DIR)/pybind/*.cpp) $(wildcard $(MODULE_SHARE_OBJS_RLT_DIR)/cg_method/*.cpp) $(wildcard $(MODULE_SHARE_OBJS_RLT_DIR)/matrix/naive/*.cpp) 
MODULE_SHARE_OBJS = $(MODULE_SHARE_OBJS_RLT_DIR)/_cgpy$(shell python3-config --extension-suffix)

#dependency
DEPS = $(TARGET:.o=.d)
-include $(DEPS)

# CPU/GPU selection in linking stage
TARGET_CPU = $(CPP_FILE_CPU:.cpp=.o)
TARGET_GPU = $(CPP_FILE_GPU:.cu=.o)
TARGET_SHARED = $(CPP_FILE_SHARED:.cpp=.o)
TARGET += $(TARGET_SHARED)
ifeq ($(GPU),1)
	LINKER = $(NVCC)
	LINK_FLAGS = $(FLAGS_GPU_LINK)
	TARGET += $(TARGET_GPU)
else
	LINKER = $(CXX)
	LINK_FLAGS = $(FLAGS_CPU)
	TARGET += $(TARGET_CPU)
endif

#Makefile
.PHONY: all demo test clean rebuild
default: all

rebuild:
	$(MAKE) all
	$(MAKE) test

all:
	make clean
	make $(MODULE_SHARE_OBJS)

$(MODULE_SHARE_OBJS): $(TARGET)
	$(LINKER) $(LINK_FLAGS) $^ -o $@

$(TARGET_CPU): %.o : %.cpp
	$(CXX) $(FLAGS_CPU) $(FLAGS_DEP) $(PYBINCLUDE) $(CXXINCLUDE_CPU)  -c $< -o $@

$(TARGET_GPU): %.o : %.cu
	$(NVCC) $(FLAGS_GPU_COMPILE) $(FLAGS_DEP) $(PYBINCLUDE) $(CXXINCLUDE_GPU)  -c $< -o $@

$(TARGET_SHARED): %.o : %.cpp
	$(CXX) $(FLAGS_CPU) $(FLAGS_DEP) $(PYBINCLUDE) $(CXXINCLUDE_CPU)  -c $< -o $@

demo: $(MODULE_SHARE_OBJS)
	mkdir -p demo/results
	python3 demo/demo_matrix.py | tee demo/results/matrix_performance.txt
	python3 demo/demo_cg_method.py | tee demo/results/cg_method_performance.txt

test: $(MODULE_SHARE_OBJS)
	python3 -m pytest -v tests/test_matrix.py
	python3 -m pytest -v tests/test_cg_method.py

clean:
	rm -rf *.so cpp/*.so cpp/*/*.so
	rm -rf cpp/*/*.o
	rm -rf */__pycache__ cpp/*/__pycache__
	rm -rf .pytest_cache */.pytest_cache
	rm -rf demo/results
	rm -rf cpp/*/*.d
	rm -rf cpp/*/*/*.d
	rm -rf cpp/*/*/*.o

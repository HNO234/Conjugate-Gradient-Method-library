WORKING_DIR = $(shell pwd)
#Flags
CXX = g++
NVCC = nvcc
FLAGS_CPU = -g -O3 -m64 -Wall -shared -std=c++17 -fPIC -fopenmp -pthread -ffast-math -ftree-vectorize
FLAGS_GPU_COMPILE = -g -O3 -m64 -Wall -shared -std=c++17 -fPIC -ffast-math -ftree-vectorize -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler
FLAGS_GPU_LINK = -g -O3 -m64 -Wall -shared -std=c++17 -fPIC -ffast-math -ftree-vectorize --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler
PYBINCLUDE = $(shell python3-config --includes) $(shell python3 -m pybind11 --includes)
FLAGS_DEP = -MMD -MP
DIRS_CPU = $(WORKING_DIR)/cpp/matrix/naive $(WORKING_DIR)/cpp/cg_method $(WORKING_DIR)/cpp/matrix/cpu $(WORKING_DIR)/cpp/pybind
CXXINCLUDE_CPU := $(patsubst %,-I %,$(DIRS_CPU))
DIRS_GPU = $(WORKING_DIR)/cpp/matrix/gpu
CXXINCLUDE_GPU := $(patsubst %,-I %,$(DIRS_GPU))
#PATH
MODULE_SHARE_OBJS_RLT_DIR = cpp
MODULE_SHARE_OBJS_ABS_DIR = $(WORKING_DIR)/$(MODULE_SHARE_OBJS_RLT_DIR)
PYTHONPATH := $(MODULE_SHARE_OBJS_ABS_DIR):$(PYTHONPATH)
export PYTHONPATH

#Includes
CPP_FILE_CPU = $(wildcard $(MODULE_SHARE_OBJS_RLT_DIR)/*/cpu/*.cpp)
CPP_FILE_GPU = $(wildcard $(MODULE_SHARE_OBJS_RLT_DIR)/*/gpu/*.cu)
CPP_FILE_SHARED = $(wildcard $(MODULE_SHARE_OBJS_RLT_DIR)/*/*.cpp) $(wildcard $(MODULE_SHARE_OBJS_RLT_DIR)/*/naive/*.cpp) 
MODULE_SHARE_OBJS = $(MODULE_SHARE_OBJS_RLT_DIR)/_cgpy$(shell python3-config --extension-suffix)
TARGET_CPU = $(CPP_FILE_CPU:.cpp=.o)
TARGET_GPU = $(CPP_FILE_GPU:.cu=.o)
TARGET_SHARED = $(CPP_FILE_SHARED:.cpp=.o)
ifeq ($(GPU),1)
	TARGET += $(TARGET_GPU)
else
	TARGET += $(TARGET_CPU)
endif
TARGET += $(TARGET_SHARED)

#dependency
DEPS = $(TARGET:.o=.d)
-include $(DEPS)

#Makefile
.PHONY: all demo test clean
default: all

rebuild:
	$(MAKE) clean
	$(MAKE) all
	$(MAKE) test

all:
	make clean
	make $(MODULE_SHARE_OBJS)

$(MODULE_SHARE_OBJS): $(TARGET)
	ifeq ($(GPU),1)
		$(NVCC) $(FLAGS_GPU_LINK) $^ -o $@
	else
		$(CXX) $(FLAGS_CPU) $^ -o $@
	endif

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

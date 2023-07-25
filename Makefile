PROJECT_ROOT=$(CURDIR)

CPPFLAGS = -I $(PROJECT_ROOT)/include/

SRC_DIR := $(PROJECT_ROOT)/src

SRCS := $(wildcard $(SRC_DIR)/*.cpp)

MAIN_FILE := $(PROJECT_ROOT)/main_seq.cpp
MAIN_PARALLEL_FILE := $(PROJECT_ROOT)/main_parallel.cpp
MAIN_CUSTOM_FILE := $(PROJECT_ROOT)/main_custom_matrix.cpp
SRCS1 := $(SRCS) $(MAIN_PARALLEL_FILE)
SRCS2 := $(SRCS) $(MAIN_FILE)
SRCS3 := $(SRCS) $(MAIN_CUSTOM_FILE)

OBJS1=$(SRCS1:.cpp=.o)
OBJS2=$(SRCS2:.cpp=.o)
OBJS3=$(SRCS3:.cpp=.o)

CXX=mpic++
CXX2=g++
CC=$(CXX)
CXXFLAGS=-O2 -Wall -std=c++17
all: main mainseq

distclean:
		$(RM) main mainseq maincustom
		$(RM) *.o

mainseq: $(SRCS2)
		$(CXX2) $(CXXFLAGS) $(SRCS2) -Wall -o mainseq $(CPPFLAGS)

main: $(SRCS1)
		$(CXX) $(CXXFLAGS) $(SRCS1)  -Wno-sign-compare -o main $(CPPFLAGS)

maincustom: $(SRCS3)
		$(CXX) $(CXXFLAGS) $(SRCS3)  -Wno-sign-compare -o maincustom $(CPPFLAGS)

run: main
ifeq ($(filter-out $@,$(MAKECMDGOALS)),)
		@echo "Usage: make run <num_processes> <testname>"
		@exit 1
endif
		mpiexec -n $(word 1, $(filter-out $@,$(MAKECMDGOALS))) ./main -t $(word 2, $(filter-out $@,$(MAKECMDGOALS)))

runseq: mainseq
ifeq ($(filter-out $@,$(MAKECMDGOALS)),)
		@echo "Usage: make runseq <testname>"
		@exit 1
endif
		./mainseq -t $(word 1, $(filter-out $@,$(MAKECMDGOALS)))

runcustom: maincustom
ifeq ($(filter-out $@,$(MAKECMDGOALS)),)
		@echo "Usage: make runcustom <num_processes>"
		@exit 1
endif
		mpiexec -n $(word 1, $(filter-out $@,$(MAKECMDGOALS))) ./maincustom 


%:
	@:

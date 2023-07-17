PROJECT_ROOT=$(CURDIR)

CPPFLAGS = -I $(PROJECT_ROOT)/include/

SRC_DIR := $(PROJECT_ROOT)/src

SRCS := $(wildcard $(SRC_DIR)/*.cpp)

MAIN_FILE := $(PROJECT_ROOT)/main.cpp
MAIN_PARALLEL_FILE := $(PROJECT_ROOT)/main_parallel.cpp
SRCS1 := $(SRCS) $(MAIN_PARALLEL_FILE)
SRCS2 := $(SRCS) $(MAIN_FILE)

OBJS1=$(SRCS1:.cpp=.o)
OBJS2=$(SRCS2:.cpp=.o)

CXX=mpic++
CXX2=g++
CC=$(CXX)
CXXFLAGS=-O2 -Wall -std=c++17
all: main mainseq

distclean:
		$(RM) main mainseq
		$(RM) *.o

mainseq: $(SRCS2)
		$(CXX2) $(CXXFLAGS) $(SRCS2) -Wall -o mainseq $(CPPFLAGS)

main: $(SRCS1)
		$(CXX) $(CXXFLAGS) $(SRCS1)  -Wno-sign-compare -o main $(CPPFLAGS)

run: main
ifeq ($(filter-out $@,$(MAKECMDGOALS)),)
		@echo "Usage: make run <num_processes> <testname>"
		@exit 1
endif
		mpiexec -n $(word 1, $(filter-out $@,$(MAKECMDGOALS))) ./main -t $(word 2, $(filter-out $@,$(MAKECMDGOALS)))

runseq: mainseq
ifeq ($(filter-out $@,$(MAKECMDGOALS)),)
		@echo "Usage: make run <testname>"
		@exit 1
endif
		./mainseq -t $(word 1, $(filter-out $@,$(MAKECMDGOALS)))


%:
	@:

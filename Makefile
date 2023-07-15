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
CXXFLAGS=-O0 -Wall -std=c++17
all: main main2

distclean:
		$(RM) main main2
		$(RM) *.o

main2: $(SRCS2)
		$(CXX2) $(CXXFLAGS) $(SRCS2) -Wall -o main2 $(CPPFLAGS)

main: $(SRCS1)
		$(CXX) $(CXXFLAGS) $(SRCS1) -Wall -o main $(CPPFLAGS)

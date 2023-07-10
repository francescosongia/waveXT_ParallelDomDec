PROJECT_ROOT=/home/scientific-vm/Desktop/pacs_primepolicy_nonfunziona_0607

CPPFLAGS = -I $(PROJECT_ROOT)/include/

SRCS := $(wildcard ./*.cpp)
FILE_SEQ := ./main.cpp
FILE_PAR := ./main_parallel.cpp
SRCS1 := $(filter-out $(FILE_SEQ),$(SRCS))
SRCS2 := $(filter-out $(FILE_PAR),$(SRCS))

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

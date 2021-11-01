SRCS = accFill_ex1.cpp 
OBJS = $(SRCS:.cpp=.o)

CC       = pgcc
CXX      = pgc++
#ACCFLAGS = -std=c++11 -acc -ta=multicore,tesla,time -Minfo=accel
#ACCFLAGS = -acc -ta=multicore,tesla,time -Minfo=accel
#ACCFLAGS = -acc -Minfo=accel
ACCFLAGS = -acc -ta=multicore,tesla,time -Minfo=accel
CXXFLAGS += -c -std=c++11 $(ACCFLAGS)

#GPUFLAGS=-ta=nvidia:cc35,cc60 -Minfo=accel,inline
GPUFLAGS=-ta=multicore,tesla,time -Minfo=accel
LDFLAGS = $(GPUFLAGS)
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJS) 
	#$(CXX) $(LDFLAGS) $^ -v -o $@
	$(CXX) $(LDFLAGS) $^ -o $@
accFill_ex1.o: accFill_ex1.cpp
	#$(CXX) $^ $(CXXFLAGS) -v -o $@
	$(CXX) $^ $(CXXFLAGS) -o $@

.PHONY: clean
clean:
	$(RM) $(TARGET)
	$(RM) $(OBJS)

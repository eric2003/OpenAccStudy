SRCS = accFill_ex2.cpp 
OBJS = $(SRCS:.cpp=.o)

CC       = pgcc
CXX      = pgc++

ACCFLAGS = -acc -ta=multicore,tesla,time -Minfo=accel
CXXFLAGS += -c -std=c++11 $(ACCFLAGS)

GPUFLAGS = -ta=multicore,tesla,time -Minfo=accel
LDFLAGS  = $(GPUFLAGS)
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $(LDFLAGS) $^ -o $@
accFill_ex2.o: accFill_ex2.cpp
	$(CXX) $^ $(CXXFLAGS) -o $@

.PHONY: clean
clean:
	$(RM) $(TARGET)
	$(RM) $(OBJS)

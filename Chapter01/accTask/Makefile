SRCS = accTask.cpp 
OBJS = $(SRCS:.cpp=.o)

CC       = pgcc
CXX      = pgc++

ACCFLAGS = -acc -ta=multicore,nvidia -Minfo=accel
CXXFLAGS += -c -std=c++11 -O3 $(ACCFLAGS)

GPUFLAGS = -ta=multicore,nvidia -Minfo=accel
LDFLAGS  = $(GPUFLAGS)
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $(LDFLAGS) $^ -o $@
accTask.o: accTask.cpp
	$(CXX) $^ $(CXXFLAGS) -o $@

.PHONY: clean
clean:
	$(RM) $(TARGET)
	$(RM) $(OBJS)

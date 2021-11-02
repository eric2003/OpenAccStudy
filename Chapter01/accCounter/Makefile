SRCS = accCounter.cpp 
OBJS = $(SRCS:.cpp=.o)

CC       = pgcc
CXX      = pgc++

ACCFLAGS = -acc -ta=multicore,nvidia -Minfo=accel
CXXFLAGS += -c -std=c++11 -fast $(ACCFLAGS)

#GPUFLAGS = -ta=multicore,nvidia:cc35 -Minfo=accel
GPUFLAGS = -ta=multicore,nvidia -Minfo=accel
LDFLAGS  = $(GPUFLAGS)
TARGET = accCounter_no_atomic

all: $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $(LDFLAGS) $^ -o $@
accCounter.o: accCounter.cpp
	$(CXX) $^ $(CXXFLAGS) -o $@

.PHONY: clean
clean:
	$(RM) $(TARGET)
	$(RM) $(OBJS)

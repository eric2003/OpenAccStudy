SRCS = accFill_ex1.cpp 
OBJS = $(SRCS:.cpp=.o)

CC       = pgcc
CXX      = pgc++
#ACCFLAGS = -std=c++11 -acc -ta=multicore,tesla,time -Minfo=accel
ACCFLAGS = -acc -ta=multicore,tesla,time -Minfo=accel
CXXFLAGS += -c -std=c++11 $(ACCFLAGS)

TARGET = main

all: $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $^ -v -o $@
accFill_ex1.o: accFill_ex1.cpp
	$(CXX) $^ $(CXXFLAGS) -v -o $@

.PHONY: clean
clean:
	$(RM) $(TARGET)
	$(RM) $(OBJS)

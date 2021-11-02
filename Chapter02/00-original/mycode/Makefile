CXX = pgc++

ACCFLAGS = -acc
GPUFLAGS = -fast -Minfo=all,ccff
ACCFLAGS += $(GPUFLAGS)

CXXFLAGS += -c -std=c++11 $(ACCFLAGS)
LDFLAGS = $(GPUFLAGS)

SRCS = main.cpp 
OBJS = $(SRCS:.cpp=.o)
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJS)
	$(info "TARGET=$(TARGET)")
	$(info "OBJS=$(OBJS)")
	${CXX} $^ -o $@ ${LDFLAGS}
main.o: main.cpp
	${CXX} $^ ${CXXFLAGS} -o $@
clean:
	rm -Rf *.o
	rm -Rf $(TARGET)

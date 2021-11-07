#
#  Copyright 2015 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

SRCS = laplace2d.cpp 
OBJS = $(SRCS:.cpp=.o)

#CC       = pgcc
CXX       = pgc++

GPUFLAGS = -ta=tesla:managed -Minfo=accel
ACCFLAGS = -acc $(GPUFLAGS)
CXXFLAGS += -fast -c -std=c++11 $(ACCFLAGS)
LDFLAGS = $(GPUFLAGS)

BIN = main

all: $(BIN)

main: $(OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@
laplace2d.o: laplace2d.cpp
	$(CXX) $^ $(CXXFLAGS) -o $@
clean:
	$(RM) $(BIN)
	$(RM) $(OBJS)

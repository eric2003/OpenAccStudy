cmake_minimum_required ( VERSION 3.15 )

project ( test )

set ( PRJ_HEADER 
	timer.h
)

add_executable( test
    laplace2d.cpp
	${PRJ_HEADER}
)

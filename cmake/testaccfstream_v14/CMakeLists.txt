cmake_minimum_required(VERSION 3.15)
set ( CMAKE_VERBOSE_MAKEFILE ON )

project ( test )

set ( PRJ_INCLUDE_DIRS )
set ( PRJ_COMPILE_DEF )
set ( PRJ_LIBRARIES )

set ( GPUFLAGS "-gpu=managed -Minfo=all,ccff -Mneginfo" )
set( ACC_FLAG "-fast -acc ${GPUFLAGS}" )

set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ACC_FLAG}" )

message ( STATUS "CMAKE_CXX_FLAGS now = ${CMAKE_CXX_FLAGS}" )

add_executable( test
    main.cpp
)

target_include_directories ( test
	PRIVATE 
        ${PRJ_INCLUDE_DIRS}
)

target_compile_definitions ( test
	PRIVATE
		${PRJ_COMPILE_DEF}
)

target_link_libraries( test
    PRIVATE
		${PRJ_LIBRARIES}
)
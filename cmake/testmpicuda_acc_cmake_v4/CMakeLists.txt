cmake_minimum_required(VERSION 3.15)
set ( CMAKE_VERBOSE_MAKEFILE ON )

project ( test )

set ( PRJ_COMPILE_DEF )
set ( CUDA_COMPILE_DEF )
set ( PRJ_LIBRARIES )

#set ( AUTO_ACC TRUE )
set ( AUTO_ACC FALSE )

find_package ( CUDAToolkit )
find_package ( OpenACC )

message ( STATUS "OpenACC_FOUND      = ${OpenACC_FOUND}" )
message ( STATUS "OpenACC_C_FOUND    = ${OpenACC_C_FOUND}" )
message ( STATUS "OpenACC_CXX_FOUND  = ${OpenACC_CXX_FOUND}" )
message ( STATUS "OpenACC_C_FLAGS    = ${OpenACC_C_FLAGS}" )
message ( STATUS "OpenACC_CXX_FLAGS  = ${OpenACC_CXX_FLAGS}" )

set ( GPUFLAGS "-gpu=managed -Minfo=all,ccff -Mneginfo" )

#if ( OpenACC_CXX_FOUND AND DEFINED AUTO_ACC )
if ( OpenACC_CXX_FOUND AND AUTO_ACC )
	list ( APPEND PRJ_COMPILE_DEF ENABLE_OPENACC )
	list ( APPEND CUDA_COMPILE_DEF ENABLE_OPENACC )	
	message ( STATUS "ENV{NVCC_COMPILER}  = $ENV{NVCC_COMPILER}" )
	if ( $ENV{NVCC_COMPILER} MATCHES "NVCC_COMPILER" )
		#set( ACC_FLAG "-fast -Minfo=accel -gpu=cc60,cc70,cc75,cc80" )
		set( ACC_FLAG "-fast ${GPUFLAGS}" )
	else()
		set( ACC_FLAG "-Ofast" )
	endif()
	if ( CUDAToolkit_FOUND )
		set ( OpenACC_CXX_FLAGS "${OpenACC_CXX_FLAGS} ${ACC_FLAG}" )
	else ()
		set ( OpenACC_CXX_FLAGS "${OpenACC_CXX_FLAGS} ${ACC_FLAG} -foffload=disable" )
	endif()
endif ()

find_package ( OpenMP )
if ( OpenMP_FOUND )
	list ( APPEND PRJ_COMPILE_DEF ENABLE_OPENMP )
	message ( STATUS "OpenMP_CXX_INCLUDE_DIRS  = ${OpenMP_CXX_INCLUDE_DIRS}" )
	message ( STATUS "OpenMP_CXX_LIBRARIES  = ${OpenMP_CXX_LIBRARIES}" )
	message ( STATUS "OpenMP_CXX_FLAGS  = ${OpenMP_CXX_FLAGS}" )
	list ( APPEND PRJ_LIBRARIES ${OpenMP_CXX_LIBRARIES} )
endif ()

set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${OpenACC_CXX_FLAGS}" )

if ( CUDAToolkit_FOUND )
	enable_language( CUDA )
	list ( APPEND PRJ_COMPILE_DEF ENABLE_CUDA )
	list ( APPEND CUDA_COMPILE_DEF ENABLE_CUDA )
	set ( CMAKE_CUDA_ARCHITECTURES "35;50;52;72;75" )
	add_library( cuda_lib STATIC 
		cuda_sub.cu
	)

	target_compile_definitions ( cuda_lib
		PRIVATE
			${CUDA_COMPILE_DEF}
	)
	
	list ( APPEND PRJ_LIBRARIES cuda_lib )
endif ()


add_executable( test
    main.cpp
	laplace2d_kernels.cpp
	myopenacc.cpp
	jacobi.cpp
)

target_compile_definitions ( test
	PRIVATE
		${PRJ_COMPILE_DEF}
)

target_link_libraries( test
    PRIVATE
		${PRJ_LIBRARIES}
)
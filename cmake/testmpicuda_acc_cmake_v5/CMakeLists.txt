cmake_minimum_required(VERSION 3.15)
set ( CMAKE_VERBOSE_MAKEFILE ON )

project ( test )

set ( PRJ_INCLUDE_DIRS )
set ( PRJ_COMPILE_DEF )
set ( CUDA_COMPILE_DEF )
set ( PRJ_LIBRARIES )

option ( MPI_ENABLE "ON for using MPI library" ON )

if ( MPI_ENABLE )
	list ( APPEND PRJ_COMPILE_DEF ENABLE_MPI )
    if ( ${CMAKE_SYSTEM_NAME} MATCHES "Windows" )
		set ( MPI_INCLUDE_DIRS "C:/Program Files (x86)/Microsoft SDKs/MPI/include" CACHE PATH "path to mpi headers" )
		set ( MPI_LIBRARIES "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/${ARCH}/msmpi.lib" CACHE PATH "path to mpi library" )
    else()
        set ( MPI_INCLUDE_DIRS "$ENV{MPI_HOME_INC}" CACHE PATH "path to mpi headers" )
        set ( MPI_LIBRARIES "$ENV{MPI_HOME_LIB}" CACHE PATH "path to mpi library" )
    endif()
	
	message ( STATUS "The MPI_LIBRARIES is ${MPI_LIBRARIES}" )
	
	list ( APPEND PRJ_LIBRARIES ${MPI_LIBRARIES} )
	list ( APPEND PRJ_INCLUDE_DIRS ${MPI_INCLUDE_DIRS} )
endif()

set ( AUTO_ACC TRUE )
#set ( AUTO_ACC FALSE )

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
		list ( APPEND PRJ_COMPILE_DEF NVCC_COMPILER_ENABLED )
	else()
		set( ACC_FLAG "-Ofast" )
		#set( ACC_FLAG "-Ofast ${GPUFLAGS}" )
	endif()
	if ( CUDAToolkit_FOUND )
		set ( OpenACC_CXX_FLAGS "${OpenACC_CXX_FLAGS} ${ACC_FLAG}" )
	else ()
		set ( OpenACC_CXX_FLAGS "${OpenACC_CXX_FLAGS} ${ACC_FLAG} -foffload=disable" )
	endif()
endif ()

message ( STATUS "OpenACC_CXX_FLAGS now = ${OpenACC_CXX_FLAGS}" )

find_package ( OpenMP )
if ( OpenMP_FOUND )
	list ( APPEND PRJ_COMPILE_DEF ENABLE_OPENMP )
	message ( STATUS "OpenMP_CXX_INCLUDE_DIRS  = ${OpenMP_CXX_INCLUDE_DIRS}" )
	message ( STATUS "OpenMP_CXX_LIBRARIES  = ${OpenMP_CXX_LIBRARIES}" )
	message ( STATUS "OpenMP_CXX_FLAGS  = ${OpenMP_CXX_FLAGS}" )
	list ( APPEND PRJ_LIBRARIES ${OpenMP_CXX_LIBRARIES} )
endif ()

set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${OpenACC_CXX_FLAGS}" )

message ( STATUS "CMAKE_CXX_FLAGS now = ${CMAKE_CXX_FLAGS}" )

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
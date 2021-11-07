# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 
CC := nvc
ACCFLAGS_1 := -fast
ACCFLAGS_2 := -fast -ta=multicore -Minfo=accel
ACCFLAGS_3 := -fast -ta=tesla:managed -Minfo=accel

BIN =  laplace_serial laplace_multicore laplace_gpu

all: $(BIN)

laplace_serial: jacobi.c laplace2d.parallel.c
	${CC} ${ACCFLAGS_1} -o laplace_serial jacobi.c laplace2d.parallel.c

laplace_multicore: jacobi.c laplace2d.parallel.c
	${CC} ${ACCFLAGS_2} -o laplace_multicore jacobi.c laplace2d.parallel.c

laplace_gpu: jacobi.c laplace2d.parallel.c
	${CC} ${ACCFLAGS_3} -o laplace_gpu jacobi.c laplace2d.kernels.c

clean:
	rm -f *.o laplace_*
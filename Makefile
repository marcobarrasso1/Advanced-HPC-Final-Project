all: jacobi

jacobi:
	mpicc \
	  -O3 \
	  -Wall -Wextra -Wpedantic \
	  -lm \
	  -Minfo=all -Mneginfo \
	  -gpu=cc80 -target=gpu -gpu=nomanaged \
	  -mp=multicore,gpu \
	  jacobi_gpu.c -o jacobi

clean:
	rm -f jacobi *.o *.x


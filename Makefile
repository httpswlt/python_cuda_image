COMMON= -DGPU -I/usr/local/cuda/include/
COMMON+= -DCUDNN
ARCH= -gencode arch=compute_52,code=[sm_52,compute_52]
CFLAGS=-Wall -Wfatal-errors -Wno-unused-result -Wno-unknown-pragmas
LDFLAGS= -lm -pthread

LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
LDFLAGS+= -lcudnn
LDFLAGS+= -lstdc++

CFLAGS+= -DGPU
CFLAGS+= -DCUDNN -I/usr/local/cudnn/include
CFLAGS+= -O2

SLIB=images.so
OBJDIR=./obj/
OBJ=images.o images_kernel.o
OBJS = $(addprefix $(OBJDIR), $(OBJ))

all:obj $(SLIB)

$(SLIB): $(OBJS)
	gcc $(COMMON) $(CFLAGS) -fPIC -shared -o $@ $^ $(LDFLAGS)

$(OBJDIR)%.o: %.c
	gcc $(COMMON) $(CFLAGS) -fpic -c $< -o $@

$(OBJDIR)%.o: %.cu
	nvcc $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -Xcompiler -fpic -o $@



obj:
	mkdir -p obj

clean:
	rm -f *.so ./obj/*.o
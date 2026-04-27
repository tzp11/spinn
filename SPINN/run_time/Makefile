# SPINN Runtime Makefile

CC = gcc
CFLAGS = -Wall -O2 -I. -I.. -g -mavx2 -mfma -fopenmp
LDFLAGS = -lm -lpthread -fopenmp

# 使用通配符收集所有算子源文件
OPS_SRCS = $(wildcard ops/activation/*.c) \
           $(wildcard ops/math/*.c) \
           $(wildcard ops/trig/*.c) \
           $(wildcard ops/conv/*.c) \
           $(wildcard ops/pool/*.c) \
           $(wildcard ops/norm/*.c) \
           $(wildcard ops/mm/*.c) \
           $(wildcard ops/shape/*.c) \
           $(wildcard ops/reduce/*.c) \
           $(wildcard ops/tensor/*.c) \
           $(wildcard ops/rnn/*.c) \
           $(wildcard ops/control/*.c)

ASM_SRCS = ops/mm/gemm_kernel_avx2_asm.S
ASM_OBJS = $(ASM_SRCS:.S=.o)

SRCS = spinn_runtime.c spinn_memory_planner.c spinn_ops.c spinn_threadpool.c main.c $(OPS_SRCS)
OBJS = $(SRCS:.c=.o)


# 目标
TARGET = spinn_run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)
	rm -f ops/*/*.o

.PHONY: all clean

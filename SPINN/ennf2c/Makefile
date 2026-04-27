# ennf2c 转译器
CC = gcc
RT = ../run_time
CFLAGS = -Wall -O2 -I. -I$(RT) -I.. -g
LDFLAGS = -lm

# 复用 runtime 模块
SRCS = ennf2c.c \
       $(RT)/spinn_runtime.c \
       $(RT)/spinn_memory_planner.c \
       ../ennf_op_types.c

TARGET = ennf2c

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean

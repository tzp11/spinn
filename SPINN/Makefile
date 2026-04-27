CC = gcc
CFLAGS = -Wall -O2 -I. -g
LDFLAGS = -lm

all: onnx2ennf

# 转换器依赖
onnx2ennf: onnx2ennf.o onnx2ennf_ops.o ennf_op_types.o onnx.proto3.pb-c.o protobuf-c.o graph_opt.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f *.o onnx2ennf *.ennf

test_convert: onnx2ennf
	./onnx2ennf mnist.simplified.onnx mnist.ennf
	ls -lh mnist.ennf

.PHONY: all clean test_convert

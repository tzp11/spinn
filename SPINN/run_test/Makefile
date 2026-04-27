CC = gcc
CFLAGS = -Wall -O2 -I. -I.. -g
LDFLAGS = -lm

all: tinyspinn

tinyspinn: tinyspinn.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test: tinyspinn
	./tinyspinn mnist.ennf

clean:
	rm -f tinyspinn

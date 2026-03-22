
CC = gcc
CFLAGS = -std=gnu99 -O3 -fopenmp -Wall

all: pi integral sharing rmm

pi: pi.c utility.h
	$(CC) $(CFLAGS) $< -o $@

integral: integral.c function.o utility.h
	$(CC) $(CFLAGS) $< -o $@

sharing: sharing.c utility.h
	$(CC) $(CFLAGS) $< -o $@

rmm: rmm.c utility.h
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f pi integral function.o sharing rmm matC.csv
CC = mpicc
CFLAGS = -std=gnu99 -O3 -Wall

TARGETS = prog0 progA progB progC rmm

all: $(TARGETS)

prog0: prog0.c function.h utility.h
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

progA: progA.c function.h utility.h
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

progB: progB.c function.h utility.h
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

progC: progC.c function.h utility.h
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

rmm: rmm.c utility.h
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)


clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
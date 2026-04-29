CC = mpicc
CFLAGS = -std=gnu99 -O3 -Wall

TARGETS = progA progB progC rmm

all: $(TARGETS)

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
CC = mpicc
CFLAGS = -std=gnu99 -O3 -Wall

TARGETS = progA progB progC progD rmm rmm_openmp_mpi

all: $(TARGETS)

progA: progA.c function.h utility.h
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

progB: progB.c function.h utility.h
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

progC: progC.c function.h utility.h
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

progD: progD.c function.h utility.h
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

rmm: rmm.c utility.h
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

rmm_openmp_mpi: rmm_openmp_mpi.c utility.h
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)


clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
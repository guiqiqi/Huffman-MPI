OBJS     = main.o
SOURCE   = main.cpp
HEADER   = huffman.h rle.h utils.h heap.h bits.h common.h
OUT      = main
CC       = mpic++
FLAGS    = -g -c -Wall
DATAFILE = encoded
LOADER   = $$(which mpirun)

all: $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT)

main.o: main.cpp
	$(CC) $(FLAGS) main.cpp

clean:
	rm -f $(OUT) $(OBJS) $(DATAFILE)

run: $(OUT)
	$(LOADER) -n 3 --host manager,worker1,worker2 ./main
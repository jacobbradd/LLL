
FLCFL = $(shell pkg-config --cflags flint)
FLLIB = $(shell pkg-config --libs flint)

INC = $(wildcard ./*.h)
CC = gcc
OPTFLAG = -O3
CFLAGS = $(OPTFLAG) $(FLCFL) -std=c2x -D_POSIX_C_SOURCE=200809L

TARGET = bench main rand_mat_int

.PHONY: all clean
all: $(TARGET)

$(TARGET): %:%.c $(INC)
	$(CC) $(CFLAGS) $< -o $@ $(FLLIB)


clean: $(TARGET)
	rm $(TARGET)

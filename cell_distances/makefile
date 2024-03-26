# Define variables
CC = gcc # Compiler
CFLAGS = -O3 -lm -fopenmp # Compiler flags
TARGET = cell_distances # Executable name
FILE = cell_distances.c # Source code script name

# Default target
.PHONY : all
all: $(TARGET)

# Compile the program
$(TARGET): $(FILE)
	$(CC) $(CFLAGS) -o $(TARGET) $(FILE)

# Clean up generated files
clean:
	rm -f $(TARGET)
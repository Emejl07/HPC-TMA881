# Define variables
CC = gcc
CFLAGS = -lm -O3 -lOpenCL #-march=native
TARGET = diffusion
SRCS = diffusion.c # List of source files

# Default target
.PHONY: all
all: $(TARGET)

# Link source files to generate the executable
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) 
	
#&& ./$(TARGET)

# Clean up generated files
clean:
	rm -f $(TARGET)

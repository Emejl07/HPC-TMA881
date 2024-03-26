# Define variables
CC = gcc
CFLAGS = -lm -O3 -lpthread #-march=native
TARGET = newton
SRCS = color_encodings.h newton.c # List of source files

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
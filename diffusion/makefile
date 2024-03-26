# Define variables
CC = gcc
LIBRARIES = -L/usr/lib64/openmpi/lib -lm -lmpi -Wl,-rpath,/usr/lib64/openmpi/lib -Wl,--enable-new-dtags
CFLAGS  = -O3 -I. -I/usr/include/openmpi-x86_64 -pthread $(LIBRARIES)
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

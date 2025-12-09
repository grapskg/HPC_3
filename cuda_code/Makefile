# Compiler
NVCC = nvcc

# Compiler Flags
# -O3: Optimization level 3
NVCC_FLAGS = -O3 -arch=sm_37

# Target Executable Name
TARGET = clahe_exec

# Build Rules
all: $(TARGET)

$(TARGET): main.o clahe.o
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) main.o clahe.o

# Compile main.cu to object file
main.o: main.cu clahe.h gputimer.h
	$(NVCC) $(NVCC_FLAGS) -c main.cu -o main.o

# Compile clahe.cu to object file
clahe.o: clahe.cu clahe.h gputimer.h
	$(NVCC) $(NVCC_FLAGS) -c clahe.cu -o clahe.o

# Clean up build files
clean:
	rm -f $(TARGET) *.o output.pgm
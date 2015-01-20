TARGET = omp
GEN = gen
all:
	gcc -std=c99 -O2 -fopenmp -o $(TARGET).out omp.c
	gcc -std=c99 -o $(GEN).out ../init/generate_table.c
run:
	./$(GEN).out $(N)
	./$(TARGET).out table$(N)x$(N).bin $(N)

all: acoustic_wave


SRCS_CU = MPI_Wave_3D.cu
EXECUTABLE=$(shell echo $(SRCS_CU) | tr A-Z a-z)
EXECUTABLE:= $(EXECUTABLE:.cu=)

#################################################
# Compiler flags
#################################################
CFLAGS += -O3
MPICC = mpicc
NVCC = nvcc
ARCH=sm_52
Dx=0
Dy=0


#################################################
# Compilation
#################################################
acoustic_wave:
	$(NVCC) -arch=$(ARCH) $(GPU_FLAGS) --compiler-bindir ${MPICC} --compiler-options $(CFLAGS) -o $(EXECUTABLE) $(SRCS_CU)

clean:
	@$(RM) $(EXECUTABLE)

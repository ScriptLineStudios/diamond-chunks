ifeq ($(OS),Windows_NT)
    XCOMPILER_FLAGS = /O2 /fp:fast
    OUTPUT = build\main.exe
else
    XCOMPILER_FLAGS = -O3,-march=native,-mtune=native,-ffast-math,-funroll-loops,-fomit-frame-pointer
    OUTPUT = build/main
endif

main: main.cu rng.cuh
	nvcc -arch=sm_75 -O3 --use_fast_math --maxrregcount=64 -Xptxas=-v -Xptxas=-O3 \
	    --ftz=true --prec-div=false --prec-sqrt=false --fmad=true \
	    --extra-device-vectorization --restrict -lineinfo \
	    -Xcompiler "$(XCOMPILER_FLAGS)" \
	    main.cu -o $(OUTPUT)
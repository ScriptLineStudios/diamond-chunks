#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#define USE_CUDA

#ifdef USE_CUDA
    #define CUDA_FUNCTION inline __device__ //__forceinline__
#else
    #define CUDA_FUNCTION
#endif

#include "rng.cuh"

typedef __align__(16) struct {
    int dx, dz, height;
    bool is_valid;
} Offset;

CUDA_FUNCTION Offset offset_new(int dx, int dz, int height) {
    return {.dx=dx, .dz=dz, .height=height, .is_valid=true};
}

CUDA_FUNCTION Offset offset_invalid_new() {
    return {.dx=-1, .dz=-1, .height=-1, .is_valid=false};
}

CUDA_FUNCTION Offset get_position_standard(RNG *rng) {
    int dx = rng_next_int(rng, 16); // spread
    int dz = rng_next_int(rng, 16);

    int i = -144;
    int j = 16;
    int plateau = 0;

    int l = ((j-i) - plateau) / 2;
    int i1 = (j-i) - l;
    int height = i + rng_next_between_inclusive(rng, 0, i1) + rng_next_between_inclusive(rng, 0, l);

    return offset_new(dx, dz, height);
}

CUDA_FUNCTION Offset get_small_diamond_position(RNG *rng, uint64_t chunk_seed) {
    // uint64_t feature_seed = rng_set_feature_seed(rng, chunk_seed, 18, 6);
    // (void)feature_seed;
    
    return get_position_standard(rng);
}

CUDA_FUNCTION Offset get_medium_diamond_position(RNG *rng, uint64_t chunk_seed) {
    // uint64_t feature_seed = rng_set_feature_seed(rng, chunk_seed, 19, 6);
    // (void)feature_seed;
    
    int dx = rng_next_int(rng, 16);
    int dz = rng_next_int(rng, 16);

    int i = -64;
    int j = -4;

    int height = rng_next_between_inclusive(rng, i, j);

    return offset_new(dx, dz, height);
}

CUDA_FUNCTION bool get_large_diamond_position(RNG *rng, uint64_t chunk_seed) {
    (void)rng_set_feature_seed(rng, chunk_seed, 20, 6);
    return (rng_next_float(rng) < 1.0F / (float)9.0);
    // idiotic branching... how stupid
    // if (!(rng_next_float(rng) < 1.0F / (float)9.0)) {
        // return offset_invalid_new();
    // }
    // return get_position_standard(rng);
}

CUDA_FUNCTION Offset get_buried_diamond_position(RNG *rng, uint64_t chunk_seed) {
    return get_position_standard(rng);
}

CUDA_FUNCTION float offset_distance_squared(const Offset *a, const Offset *b) {
    int x1 = a->dx;
    int y1 = a->height;
    int z1 = a->dz;

    int x2 = b->dx;
    int y2 = b->height;
    int z2 = b->dz;

    return ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1));
}

CUDA_FUNCTION bool in_range(int y) {
    return (y > -55) && (y < -6);
}

CUDA_FUNCTION bool get_small_diamond_offsets(RNG *rng, uint64_t chunk_seed, Offset *offsets, size_t *sz) {
    uint64_t feature_seed = rng_set_feature_seed(rng, chunk_seed, 18, 6);
    Offset o = get_small_diamond_position(rng, chunk_seed);
    // which idiot wrote this?? possible wasteful memory writes... (looks like it might not be that impactful)
    offsets[*sz] = o;
    (*sz)++;
    return in_range(o.height);
}

CUDA_FUNCTION bool get_medium_diamond_offsets(RNG *rng, uint64_t chunk_seed, Offset *offsets, size_t *sz) {
    uint64_t feature_seed = rng_set_feature_seed(rng, chunk_seed, 19, 6);
    Offset o = get_medium_diamond_position(rng, chunk_seed);
    offsets[*sz] = o;
    (*sz)++;
    return in_range(o.height);
}

CUDA_FUNCTION bool get_buried_diamond_offsets(RNG *rng, uint64_t chunk_seed, Offset *offsets, size_t *sz) {
    uint64_t feature_seed = rng_set_feature_seed(rng, chunk_seed, 21, 6);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        Offset o = get_buried_diamond_position(rng, chunk_seed);
        
        if (!in_range(o.height)) {
            return false;
        }

        offsets[*sz] = o;
        (*sz)++;

        rng_next_float(rng);
        rng_next_int(rng, 3);
        rng_next_int(rng, 3);

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            rng_next_double(rng);
        }
    }
    return true;
}

__managed__ unsigned long long int results = 0;

__global__ void kernel(uint64_t s, uint64_t *out) {
    uint64_t chunk_seed = blockDim.x * blockIdx.x + threadIdx.x + s;

    RNG rng = rng_new();
    
    if (!get_large_diamond_position(&rng, chunk_seed)) {
        return;
    }

    Offset offsets[15] = {0};
    size_t sz = 1;
    offsets[0] = get_position_standard(&rng); // large

    if (!get_small_diamond_offsets(&rng, chunk_seed, offsets, &sz) || 
        !get_medium_diamond_offsets(&rng, chunk_seed, offsets, &sz) || 
        !get_buried_diamond_offsets(&rng, chunk_seed, offsets, &sz)
    ) {
        return;
    }

    const Offset *cmp = &offsets[0];
    #pragma unroll
    for (int i = 1; i < 6; i++) {
        if (offset_distance_squared(cmp, (const Offset *)&offsets[i]) > 20.0) {
            return;
        }
    }

    out[atomicAdd(&results, 1ull)] = chunk_seed;
}

int main(int argc, char **argv) {
    if (argc <= 1) {
        printf("Run with the following arguments: <start block> <end block>\n");
        exit(1);
    }

    const uint64_t seed_start = atoll(argv[1]);
    const uint64_t seed_end = atoll(argv[2]);

	const uint64_t blocks = 16777216;
	const uint64_t threads = 256;

    FILE *output = fopen("results.txt", "w");

    uint64_t *buffer;
    cudaMallocManaged(&buffer, sizeof(uint64_t) * 640000); // I can't envision a world where this overflows...

    for (uint64_t chunk = seed_start; chunk < seed_end; chunk += 1000) {
        printf("working on blocks %lu to %lu\n", chunk, chunk + 1000);
        for (uint64_t s = chunk; s < chunk + 1000; s++) {
            kernel<<<blocks, threads>>>(blocks * threads * s, buffer);
        }
        cudaDeviceSynchronize();
        printf("\tfound %llu results\n", results);
        for (unsigned long long int i = 0; i < results; i++) {
            fprintf(output, "%lu\n", buffer[i]);
        }
        fflush(output);
        results = 0ull;   
    }

    fclose(output);

    return 0;
}

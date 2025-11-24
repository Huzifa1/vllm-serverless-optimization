#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

__global__ void initKernel(int *data, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        data[0] = value;
    }
}

int main() {
    // 1. Allocate device memory
    int *d_ptr = nullptr;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMalloc(&d_ptr, sizeof(int)));

    // 2. Initialize it on the GPU
    initKernel<<<1, 1>>>(d_ptr, 42);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 3. Get IPC memory handle
    cudaIpcMemHandle_t handle;
    CHECK_CUDA(cudaIpcGetMemHandle(&handle, d_ptr));

    // 4. Write handle to a file (simple IPC transport)
    FILE *f = fopen("ipc_handle.bin", "wb");
    if (!f) {
        perror("fopen");
        return EXIT_FAILURE;
    }
    size_t written = fwrite(&handle, 1, sizeof(handle), f);
    fclose(f);
    if (written != sizeof(handle)) {
        fprintf(stderr, "Failed to write full handle\n");
        return EXIT_FAILURE;
    }

    printf("Process A: wrote IPC handle. PID=%d\n", getpid());
    printf("Now run process B.\n");

    // 5. Optionally wait before exiting, so memory stays valid
    //    In a real app, you'd coordinate via proper IPC/sync.
    printf("Press Enter to exit Process A...\n");
    getchar();

    // 6. Clean up
    CHECK_CUDA(cudaFree(d_ptr));
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}

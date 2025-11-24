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

__global__ void incrementKernel(int *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        data[0] += 1;
    }
}

int main() {
    // 1. Read IPC handle from file
    cudaIpcMemHandle_t handle;
    FILE *f = fopen("ipc_handle.bin", "rb");
    if (!f) {
        perror("fopen");
        return EXIT_FAILURE;
    }
    size_t read = fread(&handle, 1, sizeof(handle), f);
    fclose(f);
    if (read != sizeof(handle)) {
        fprintf(stderr, "Failed to read full handle\n");
        return EXIT_FAILURE;
    }

    // 2. Open the handle as a device pointer
    CHECK_CUDA(cudaSetDevice(0));
    int *d_ptr = nullptr;
    CHECK_CUDA(cudaIpcOpenMemHandle(
        (void**)&d_ptr,
        handle,
        cudaIpcMemLazyEnablePeerAccess
    ));

    // 3. Read the value from GPU, print it
    int value = 0;
    CHECK_CUDA(cudaMemcpy(&value, d_ptr, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Process B: initial value from shared GPU memory = %d\n", value);

    // 4. Modify it on the GPU
    incrementKernel<<<1, 1>>>(d_ptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5. Read back and print again
    CHECK_CUDA(cudaMemcpy(&value, d_ptr, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Process B: value after increment = %d\n", value);

    // 6. Close the IPC handle (does NOT free the memory in Process A)
    CHECK_CUDA(cudaIpcCloseMemHandle(d_ptr));
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}

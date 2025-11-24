#include <cstdio>
#include <cstdlib>
#include <unistd.h>          // getpid, sleep
#include <sys/stat.h>        // stat
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

__global__ void incrementKernel(int *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        data[0] += 1;
    }
}

// simple helper: wait until a file exists
void wait_for_file(const char *path) {
    struct stat st;
    while (stat(path, &st) != 0) {
        usleep(100000); // 100 ms
    }
}

int main() {
    printf("Process A (PID %d): starting\n", getpid());
    CHECK_CUDA(cudaSetDevice(0));

    // 1. Allocate device memory and initialize
    int *d_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ptr, sizeof(int)));
    initKernel<<<1, 1>>>(d_ptr, 100);  // start with 100
    CHECK_CUDA(cudaDeviceSynchronize());

    // 2. Export IPC memory handle
    cudaIpcMemHandle_t memHandle;
    CHECK_CUDA(cudaIpcGetMemHandle(&memHandle, d_ptr));

    FILE *f = fopen("mem_handle.bin", "wb");
    if (!f) {
        perror("fopen mem_handle.bin");
        return EXIT_FAILURE;
    }
    if (fwrite(&memHandle, 1, sizeof(memHandle), f) != sizeof(memHandle)) {
        fprintf(stderr, "Process A: failed to write mem handle\n");
        fclose(f);
        return EXIT_FAILURE;
    }
    fclose(f);
    printf("Process A: wrote mem_handle.bin\n");

    // 3. Wait for B to create an interprocess event and export it
    printf("Process A: waiting for event_handle.bin from B...\n");
    wait_for_file("event_handle.bin");

    cudaIpcEventHandle_t evtHandle;
    f = fopen("event_handle.bin", "rb");
    if (!f) {
        perror("fopen event_handle.bin");
        return EXIT_FAILURE;
    }
    if (fread(&evtHandle, 1, sizeof(evtHandle), f) != sizeof(evtHandle)) {
        fprintf(stderr, "Process A: failed to read event handle\n");
        fclose(f);
        return EXIT_FAILURE;
    }
    fclose(f);
    printf("Process A: got event handle from B\n");

    // 4. Open the interprocess event
    cudaEvent_t evtFromB;
    CHECK_CUDA(cudaIpcOpenEventHandle(&evtFromB, evtHandle));

    // 5. Wait for B to signal "ready"
    printf("Process A: waiting on CUDA event from B...\n");
    CHECK_CUDA(cudaEventSynchronize(evtFromB));
    printf("Process A: event from B received\n");

    // 6. Read the value, increment it on GPU, read again
    int value = 0;
    CHECK_CUDA(cudaMemcpy(&value, d_ptr, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Process A: value after B finished = %d\n", value);

    incrementKernel<<<1, 1>>>(d_ptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&value, d_ptr, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Process A: value after A increment = %d\n", value);

    // 7. Signal B that A is done using the event/memory
    FILE *fdone = fopen("a_done.bin", "wb");
    if (fdone) {
        const char msg[] = "OK";
        fwrite(msg, 1, sizeof(msg), fdone);
        fclose(fdone);
        printf("Process A: signaled B via a_done.bin\n");
    } else {
        perror("fopen a_done.bin");
    }

    // 8. Cleanup
    CHECK_CUDA(cudaEventDestroy(evtFromB));
    CHECK_CUDA(cudaFree(d_ptr));
    CHECK_CUDA(cudaDeviceReset());

    printf("Process A: done\n");
    return 0;
}

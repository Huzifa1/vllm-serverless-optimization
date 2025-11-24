#include <cstdio>
#include <cstdlib>
#include <unistd.h>          // getpid, usleep
#include <sys/stat.h>
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
        data[0] += 10;   // B will add 10
    }
}

void wait_for_file(const char *path) {
    struct stat st;
    while (stat(path, &st) != 0) {
        usleep(100000); // 100 ms
    }
}

int main() {
    printf("Process B (PID %d): starting\n", getpid());
    CHECK_CUDA(cudaSetDevice(0));

    // 1. Wait for A to write the memory handle
    printf("Process B: waiting for mem_handle.bin from A...\n");
    wait_for_file("mem_handle.bin");

    cudaIpcMemHandle_t memHandle;
    FILE *f = fopen("mem_handle.bin", "rb");
    if (!f) {
        perror("fopen mem_handle.bin");
        return EXIT_FAILURE;
    }
    if (fread(&memHandle, 1, sizeof(memHandle), f) != sizeof(memHandle)) {
        fprintf(stderr, "Process B: failed to read mem handle\n");
        fclose(f);
        return EXIT_FAILURE;
    }
    fclose(f);
    printf("Process B: got mem handle from A\n");

    // 2. Open the shared memory
    int *d_ptr = nullptr;
    CHECK_CUDA(cudaIpcOpenMemHandle(
        (void**)&d_ptr, memHandle, cudaIpcMemLazyEnablePeerAccess));

    // 3. Create an interprocess event and export its handle
    cudaEvent_t readyEvent;
    CHECK_CUDA(cudaEventCreateWithFlags(
        &readyEvent, cudaEventDisableTiming | cudaEventInterprocess));

    cudaIpcEventHandle_t evtHandle;
    CHECK_CUDA(cudaIpcGetEventHandle(&evtHandle, readyEvent));

    f = fopen("event_handle.bin", "wb");
    if (!f) {
        perror("fopen event_handle.bin");
        return EXIT_FAILURE;
    }
    if (fwrite(&evtHandle, 1, sizeof(evtHandle), f) != sizeof(evtHandle)) {
        fprintf(stderr, "Process B: failed to write event handle\n");
        fclose(f);
        return EXIT_FAILURE;
    }
    fclose(f);
    printf("Process B: wrote event_handle.bin for A\n");

    // 4. Do some work on shared memory
    int value = 0;
    CHECK_CUDA(cudaMemcpy(&value, d_ptr, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Process B: initial value read = %d\n", value);

    incrementKernel<<<1, 1>>>(d_ptr);
    CHECK_CUDA(cudaEventRecord(readyEvent, 0));  // event will complete after kernel
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&value, d_ptr, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Process B: value after B increment = %d\n", value);

    printf("Process B: waiting for A to finish using the event...\n");
    wait_for_file("a_done.bin");
    printf("Process B: detected a_done.bin, cleaning up\n");

    // 5. Cleanup (now safe)
    CHECK_CUDA(cudaIpcCloseMemHandle(d_ptr));
    CHECK_CUDA(cudaEventDestroy(readyEvent));
    CHECK_CUDA(cudaDeviceReset());

    printf("Process B: done\n");
    return 0;
}

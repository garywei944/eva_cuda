/*
 * Copyright (c) 2026 Gary Wei
 *
 * Licensed under the MIT
 * See LICENSE file in the project root for full license information.
 */

#include <stdio.h>

__global__ void whoami(void) {
  int bid =
      blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  int block_offset = bid * blockDim.x * blockDim.y * blockDim.z;
  int thread_offset = threadIdx.x + threadIdx.y * blockDim.x +
                      threadIdx.z * blockDim.x * blockDim.y;
  int tid = block_offset + thread_offset;
  printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n", tid,
         blockIdx.x, blockIdx.y, blockIdx.z, bid, threadIdx.x, threadIdx.y,
         threadIdx.z, thread_offset);
}

int main(int argc, char **argv) {
  const int b_x = 2, b_y = 3, b_z = 4;
  const int t_x = 4, t_y = 4, t_z = 4;

  dim3 blocksPerGrid(b_x, b_y, b_z);
  dim3 threadsPerBlocks(t_x, t_y, t_z);

  whoami<<<blocksPerGrid, threadsPerBlocks>>>();
  cudaDeviceSynchronize();

  return 0;
}

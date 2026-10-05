# cuda-gdb -x leetgpu/softmax/softmax.gdb build64_debug/leetgpu_softmax_softmax
#
# Stops in block 0 / thread 0 at each stage of the kernel; `continue` moves on.

set cuda break_on_launch none
# Kernel code is loaded only after the program starts.
set breakpoint pending on

# 1. Block 0 is about to write its partial to global_buf[0] (`next` to write it).
#    Not line 132: the inlined grid.sync() there never triggers a breakpoint.
break softmax.cu:129 if blockIdx.x == 0 && threadIdx.x == 0

# 2. After the grid.sync: every block's partial is in global_buf.
break softmax.cu:134 if blockIdx.x == 0 && threadIdx.x == 0

# 3. The merged (max, sum) each thread uses for its output.
break softmax.cu:138 if blockIdx.x == 0 && threadIdx.x == 0

# partials: print every block's (max, sum) in global_buf
define partials
  print global_buf[0]@gridDim.x
end
document partials
Print every block's (max, sum) partial: global_buf[0]@gridDim.x.
end

run

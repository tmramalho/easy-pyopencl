import pyopencl as cl
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
n_threads = ctx.get_info(cl.context_info.DEVICES)[0].max_work_group_size
n_items = 2*n_threads

a = np.arange(n_items).astype(np.float32)
r = np.empty(n_items).astype(np.float32)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
r_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=r.nbytes)
loc_buf = cl.LocalMemory(4*n_items)

prg = cl.Program(ctx, """
    __kernel void scan(__global float *a,
            __global float *r,
            __local float *b,
            uint n_items)
    {
      uint gid = get_global_id(0);
      uint lid = get_local_id(0);
      uint dp = 1;

      b[2*lid] = a[2*gid];
      b[2*lid+1] = a[2*gid+1];

      for(uint s = n_items>>1; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid < s) {
          uint i = dp*(2*lid+1)-1;
          uint j = dp*(2*lid+2)-1;
          b[j] += b[i];
        }

        dp <<= 1;
      }

      if(lid == 0) b[n_items - 1] = 0;

      for(uint s = 1; s < n_items; s <<= 1) {
        dp >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if(lid < s) {
          uint i = dp*(2*lid+1)-1;
          uint j = dp*(2*lid+2)-1;

          float t = b[i];
          b[i] = b[j];
          b[j] += t;
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      r[2*gid] = b[2*lid];
      r[2*gid+1] = b[2*lid+1];
    }
    """).build()

evt = prg.scan(queue, (n_threads,), None, a_buf, r_buf, loc_buf, np.uint32(n_items))
evt.wait()
print evt.profile.end - evt.profile.start

cl.enqueue_copy(queue, r, r_buf)
ex_scan = np.pad(np.cumsum(a), (1,0), mode ='constant')[:-1]
print np.allclose(r,ex_scan)
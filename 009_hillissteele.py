import pyopencl as cl
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
n_threads = ctx.get_info(cl.context_info.DEVICES)[0].max_work_group_size

a = np.arange(n_threads).astype(np.float32)
r = np.empty(n_threads).astype(np.float32)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
r_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=r.nbytes)
loc_buf = cl.LocalMemory(4*n_threads)

prg = cl.Program(ctx, """
    #define SWAP(a,b) {__local float *tmp=a;a=b;b=tmp;}
    __kernel void scan(__global float *a,
            __global float *r,
            __local float *b,
            __local float *c)
    {
      uint gid = get_global_id(0);
      uint lid = get_local_id(0);
      uint gs = get_local_size(0);

      c[lid] = b[lid] = a[gid];
      barrier(CLK_LOCAL_MEM_FENCE);

      for(uint s = 1; s < gs; s <<= 1) {
        if(lid > (s-1)) {
          c[lid] = b[lid]+b[lid-s];
        } else {
          c[lid] = b[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(b,c);
      }
      r[gid] = b[lid];
    }
    """).build()

evt = prg.scan(queue, (n_threads,), None, a_buf, r_buf, loc_buf, loc_buf)
evt.wait()
print evt.profile.end - evt.profile.start

cl.enqueue_copy(queue, r, r_buf)
print np.allclose(r,np.cumsum(a))
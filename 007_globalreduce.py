import pyopencl as cl
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
n_threads = ctx.get_info(cl.context_info.DEVICES)[0].max_work_group_size

N = n_threads*n_threads
a = np.arange(N).astype(np.float32)
r = np.empty(n_threads).astype(np.float32)
o = np.empty(1).astype(np.float32)
print "Reducing {0:d} numbers...".format(N)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
r_buf = cl.Buffer(ctx, mf.READ_WRITE, size=r.nbytes)
o_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=o.nbytes)

prg = cl.Program(ctx, """
    __kernel void reduce(__global float *a, __global float *r)
    {
      uint gid = get_global_id(0);
      uint wid = get_group_id(0);
      uint lid = get_local_id(0);
      uint gs = get_local_size(0);

      for(uint s = gs/2; s > 0; s >>= 1) {
        if(lid < s) {
          a[gid] += a[gid+s];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
      }
      if(lid == 0) r[wid] = a[gid];
    }
    """).build()

evt = prg.reduce(queue, (N,), (n_threads,), a_buf, r_buf)
evt.wait()
print evt.profile.end - evt.profile.start

evt = prg.reduce(queue, (n_threads,), (n_threads,), r_buf, o_buf)
evt.wait()
print evt.profile.end - evt.profile.start
cl.enqueue_copy(queue, o, o_buf)
print o, np.sum(a)
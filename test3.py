import pyopencl as cl
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
n_threads = ctx.get_info(cl.context_info.DEVICES)[0].max_work_group_size
n_groups = ctx.get_info(cl.context_info.DEVICES)[0].max_compute_units

a = np.zeros(n_groups).astype(np.int32)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)

prg = cl.Program(ctx, """
    __kernel void hist(__global int *a, int n_groups)
    {
      int wid = get_group_id(0);
      int gr = wid % n_groups;

      atomic_add(a+gr, gr);
    }
    """).build()

evt = prg.hist(queue, (100*n_groups*n_threads,), None, a_buf, np.int32(n_groups))
evt.wait()
print evt.profile.end - evt.profile.start

cl.enqueue_copy(queue, a, a_buf)
print a, n_groups, n_threads
import pyopencl as cl
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
n_threads = ctx.get_info(cl.context_info.DEVICES)[0].max_work_group_size
n_items = 2**16
n_bins = 2**6
n_groups = ctx.get_info(cl.context_info.DEVICES)[0].max_compute_units
n_pad_groups = 2**(n_groups-1).bit_length()
n_runs = n_threads*n_groups
n_batch = np.ceil(n_items/float(n_runs))

max_local_size = ctx.get_info(cl.context_info.DEVICES)[0].local_mem_size
local_size = 4*n_bins
if local_size > max_local_size:
	print "Too many bins!"
	exit()

a = np.arange(n_items).astype(np.int32)
np.random.shuffle(a)
b = np.zeros(n_pad_groups*n_bins).astype(np.int32)
r = np.empty(n_bins).astype(np.int32)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=b)
r_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=r.nbytes)
loc_buf = cl.LocalMemory(local_size)

prg = cl.Program(ctx, """
    __kernel void hist(__global int *a,
            __global int *b,
            __local int *h, uint n_batch,
            uint n_bins, uint n_items,
            uint max_val)
    {
      uint gid = get_global_id(0);
      uint wid = get_group_id(0);
      uint lid = get_local_id(0);
      uint ls = get_local_size(0);

      for(uint s = 0; s < n_bins; s+= ls) {
        uint pos = lid+s;
        if(pos < n_bins) {
          h[pos] = b[n_bins*wid+pos];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      for(uint n = 0; n < n_batch; n++) {
        uint pos = gid*n_batch + n;
        if(pos < n_items) {
          int bin_pos = clamp(n_bins * a[pos] / max_val, (uint)0, n_bins-1);
          atomic_inc(h+bin_pos);
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      for(uint s = 0; s < n_bins; s+= ls) {
        uint pos = lid+s;
        if(pos < n_bins) {
          b[n_bins*wid+pos] = h[pos];
        }
      }
    }
    __kernel void red_bins(__global int *b, __global int *r, uint n_bins)
    {
      uint gid = get_global_id(0);
      uint lid = get_local_id(0);
      uint gs = get_local_size(0);

      for(uint s = gs/2; s > 0; s >>= 1) {
        if(lid < s) {
          for(uint bs = 0; bs < n_bins; bs++) {
            b[gid*n_bins+bs] += b[(gid+s)*n_bins+bs];
          }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
      }
      if(lid == 0) {
        for(uint bs = 0; bs < n_bins; bs++) {
            r[bs] = b[gid*n_bins+bs];
          }
      }
    }
    """).build()

evt = prg.hist(queue, (n_runs,), None, a_buf, b_buf, loc_buf,
               np.uint32(n_batch), np.uint32(n_bins), np.uint32(n_items), np.uint32(1E5))
evt.wait()
print evt.profile.end - evt.profile.start

cl.enqueue_copy(queue, b, b_buf)

evt = prg.red_bins(queue, (n_pad_groups,), None, b_buf, r_buf, np.uint32(n_bins))
evt.wait()
print evt.profile.end - evt.profile.start

cl.enqueue_copy(queue, r, r_buf)
print r, r.sum()

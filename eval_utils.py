import numpy as np
import loopy as lp


# {{{ get_average_exec_time

def get_average_exec_time(
        ctx, queue, knl, knl_arg_dict,
        n_trial_sets=4, n_wtime_trials=7, n_warmup_wtime_trials=3):
    import time

    # from loopy.target.pyopencl import adjust_local_temp_var_storage
    # knl = adjust_local_temp_var_storage(knl, queue.device)
    # knl = lp.set_options(knl, no_numpy=True)
    compiled = lp.CompiledKernel(ctx, knl)

    avg_times = []
    for ts in range(n_trial_sets):
        trial_wtimes = []
        for t in range(n_wtime_trials + n_warmup_wtime_trials):
            queue.finish()
            tstart = time.time()
            evt, out = compiled(queue, **knl_arg_dict)
            queue.finish()
            tend = time.time()
            trial_wtimes.append(tend-tstart)

        avg_times.append(np.average(trial_wtimes[n_warmup_wtime_trials:]))

    min_avg_times = min(avg_times)
    return min_avg_times

# }}}


# {{{ get_workgroup_size

def get_workgroup_size(knl):
    from loopy.symbolic import aff_to_expr
    global_size, local_size = knl.get_grid_size_upper_bounds()
    workgroup_size = 1
    if local_size:
        for size in local_size:
            if size.n_piece() != 1:
                raise ValueError(
                    "Workgroup size found to be genuinely "
                    "piecewise defined, which is not allowed in stats gathering")

            (valid_set, aff), = size.get_pieces()

            assert ((valid_set.n_basic_set() == 1)
                    and (valid_set.get_basic_sets()[0].is_universe()))

            s = aff_to_expr(aff)
            if not isinstance(s, int):
                raise ValueError(
                    "work-group size is not integer: %s"
                    % (local_size))
            workgroup_size *= s

    return global_size, local_size, workgroup_size

# }}}


# {{{ get_op_ct

def get_op_ct(knl, dtype_list, param_dict=None, count_redundant_work=True):
    _, _, threads_per_group = get_workgroup_size(knl)
    effective_sgs = min(32, threads_per_group)
    from loopy.statistics import get_op_map
    op_map = get_op_map(
        knl,
        count_redundant_work=count_redundant_work,
        count_within_subscripts=False,
        subgroup_size=effective_sgs,
        # count_madds=self.count_madds,
        )

    # flops counted w/subgroup granularity
    # (multiply count by size of subgroup)
    total_ops = op_map.filter_by(
        dtype=dtype_list,
        count_granularity=[lp.CountGranularity.SUBGROUP],
        ).sum() * effective_sgs

    # flops counted w/workitem granularity (should be zero)
    total_ops += op_map.filter_by(
        dtype=dtype_list,
        count_granularity=[lp.CountGranularity.WORKITEM],
        ).sum()

    # NOTE if madds are being counted as 1, need to count them twice

    if param_dict is not None:
        return total_ops.eval_with_dict(param_dict)
    else:
        return total_ops

# }}}


# {{{ get_global_data_moved

def get_global_data_moved(
        knl, param_dict=None, count_redundant_work=True, variables=None):
    _, _, threads_per_group = get_workgroup_size(knl)
    effective_sgs = min(32, threads_per_group)
    from loopy.statistics import get_mem_access_map
    mem_access_map = get_mem_access_map(
        knl,
        count_redundant_work=count_redundant_work,
        subgroup_size=effective_sgs,
        )
    print("MEM ACCESS MAP-----------------------")
    print(mem_access_map)
    print("-------------------------------------")

    if variables is not None:
        mem_access_map = mem_access_map.filter_by(variable=variables)

    from loopy import gather_access_footprint_bytes
    footsize_bytes = 0
    for access, count in mem_access_map.filter_by(
                mtype=["global"]).items():
        direction = "write" if access.direction == "store" else "read"
        footsize_bytes += gather_access_footprint_bytes(knl)[
                (access.variable, direction)].eval_with_dict(param_dict)

    # mem access counted w/subgroup granularity
    # (multiply count by size of subgroup)
    data_moved_bytes = mem_access_map.filter_by(
        mtype=["global"],
        count_granularity=[lp.CountGranularity.SUBGROUP],
        ).to_bytes().sum()*effective_sgs
    # mem access counted w/workitem granularity
    data_moved_bytes += mem_access_map.filter_by(
        mtype=["global"],
        count_granularity=[lp.CountGranularity.WORKITEM],
        ).to_bytes().sum()

    if param_dict is not None:
        data_moved_bytes = data_moved_bytes.eval_with_dict(param_dict)

    # return (data_moved_bytes, footsize_bytes)
    return data_moved_bytes, footsize_bytes

# }}}


# {{{ get_wtime

def get_wtime(ctx, queue, knl, knl_arg_dict, consistency_check=False):
    wtime = get_average_exec_time(ctx, queue, knl, knl_arg_dict)

    # sanity check to make sure timing strategy yields consistent times:
    if consistency_check:
        consistency_check_times = []
        for i in range(5):
            consistency_check_times.append(
                get_average_exec_time(ctx, queue, knl, knl_arg_dict))
        rel_stddev = (
            np.std(consistency_check_times)/np.mean(consistency_check_times)
            )/100.0

        if rel_stddev > 0.01:
            raise Exception(
                "inconsistent execution times, "
                "relative standard deviation is too high")

        return wtime, rel_stddev
    else:
        return wtime, None

# }}}


# {{{ print_stats

def print_stats(knl, param_dict, wtime, flop_dtype, data_moved_est=None):
    # make sure barvinok is installed
    import islpy as isl
    try:
        isl.BasicSet.card
    except AttributeError:
        # raise Exception("BARVINOK NOT INSTALLED")
        print("BARVINOK NOT INSTALLED!!!!")

    global_size, local_size, threads_per_group = get_workgroup_size(knl)
    print("Global, local sizes:")
    print(global_size)
    print("%s (%s threads)" % (local_size, threads_per_group))

    giga = 10**9

    flop_ct = get_op_ct(knl, dtype_list=[flop_dtype], param_dict=param_dict)
    flop_rate = flop_ct/wtime

    data_moved_bytes, footsize_bytes = get_global_data_moved(
        knl, param_dict=param_dict)
    PEAK_BANDWIDTH_TitanV = 652.8*giga  # noqa
    PEAK_FLOPS32_TitanV = 12288*giga  # noqa
    PEAK_FLOPS64_TitanV = 6144*giga  # noqa
    import numpy as np
    assert flop_dtype in [np.float32, np.float64]
    peak_flops = PEAK_FLOPS32_TitanV if (
        flop_dtype == np.float32) else PEAK_FLOPS64_TitanV

    print("time:", wtime)
    print("64-bit flops:", flop_ct)
    print("flop/s:", flop_rate)
    print("flop/s percent of peak (Titan V): %f%%" % (
        flop_rate/peak_flops*100))

    mem_stats = [
            ("all data accessed", data_moved_bytes),
            ("data footprint", footsize_bytes),
            ]
    if data_moved_est is not None:
        mem_stats.append(("true data estimate", data_moved_est))
    for descr, data_size in mem_stats:
        throughput = data_size/wtime
        print("--------- %s ---------" % (descr))
        print("%s: %f GB" % (descr, data_size/giga))
        print("throughput: %f GB/s" % (throughput/giga))
        print("throughput percent of peak (Titan V): %f%%" % (
            throughput/PEAK_BANDWIDTH_TitanV*100))

    return flop_rate, throughput, data_moved_bytes, footsize_bytes

# }}}


# {{{ write_to_out_dir

def write_to_out_dir(filename, contents):
    fname_with_dir = "out/%s" % (filename)
    import os
    if not os.path.exists(os.path.dirname(fname_with_dir)):
        os.makedirs(os.path.dirname(fname_with_dir))
    with open(fname_with_dir, 'w') as f:
        print(contents, file=f)

# }}}


# {{{ check_for_nans

def check_for_nans(array, print_indices=True):
    import sys
    nan_indices = np.argwhere(np.isnan(array))
    nan_ct = np.count_nonzero(np.isnan(array))
    np.set_printoptions(threshold=sys.maxsize)
    if nan_ct > 0:
        print("NaNs FOUND!!!!")
        if print_indices:
            print("NaN locations:\n", nan_indices)
        print("NaN counts:", nan_ct)
        return True
    return False

# }}}


# {{{ get_jacld_knl_str

def get_jacld_knl_str(i_iname="i"):

    # {{{ knl string
    s = """
        <>r43 = 4.0/3.0  {{id=s0_temp_init}}
        <>c1345 = c1*c3*c4*c5  {{id=s1_temp_init}}
        <>c34 = c3*c4  {{id=s2_temp_init}}
        for {0}
            <>tmp1 = rho_i[1+{0}, 1+j, 1+k]  {{id=s3}}
            <>tmp2 = tmp1*tmp1  {{id=s4}}
            <>tmp3 = tmp1*tmp2  {{id=s5}}

            <>d[0, 0, {0}] = 1+dt*2*(tx1*dx1+ty1*dy1+tz1*dz1)  {{id=s6_write_d_i}}
            d[0, 1, {0}] = 0  {{id=s7_write_d_i}}
            d[0, 2, {0}] = 0  {{id=s8_write_d_i}}
            d[0, 3, {0}] = 0  {{id=s9_write_d_i}}
            d[0, 4, {0}] = 0  {{id=s10_write_d_i}}

            d[1, 0, {0}] = -dt*2*(tx1*r43+ty1+tz1)*c34*tmp2*u[1, 1+{0}, 1+j, 1+k]  {{id=s11_write_d_i}}
            d[1, 1, {0}] = 1+dt*2*c34*tmp1*(tx1*r43+ty1+tz1)+dt*2*(tx1*dx2+ty1*dy2+tz1*dz2)  {{id=s12_write_d_i}}
            d[1, 2, {0}] = 0  {{id=s13_write_d_i}}
            d[1, 3, {0}] = 0  {{id=s14_write_d_i}}
            d[1, 4, {0}] = 0  {{id=s15_write_d_i}}

            d[2, 0, {0}] = -dt*2*(tx1+ty1*r43+tz1)*c34*tmp2*u[2, 1+{0}, 1+j, 1+k]  {{id=s16_write_d_i}}
            d[2, 1, {0}] = 0  {{id=s17_write_d_i}}
            d[2, 2, {0}] = 1+dt*2*c34*tmp1*(tx1+ty1*r43+tz1)+dt*2*(tx1*dx3+ty1*dy3+tz1*dz3)  {{id=s18_write_d_i}}
            d[2, 3, {0}] = 0  {{id=s19_write_d_i}}
            d[2, 4, {0}] = 0  {{id=s20_write_d_i}}

            d[3, 0, {0}] = -dt*2*(tx1+ty1+tz1*r43)*c34*tmp2*u[3, 1+{0}, 1+j, 1+k]  {{id=s21_write_d_i}}
            d[3, 1, {0}] = 0  {{id=s22_write_d_i}}
            d[3, 2, {0}] = 0  {{id=s23_write_d_i}}
            d[3, 3, {0}] = 1+dt*2*c34*tmp1*(tx1+ty1+tz1*r43)+dt*2*(tx1*dx4+ty1*dy4+tz1*dz4)  {{id=s24_write_d_i}}
            d[3, 4, {0}] = 0  {{id=s25_write_d_i}}

            d[4, 0, {0}] = -dt*2*( \
                ( \
                    (tx1*(r43*c34-c1345)+ty1*(c34-c1345)+tz1*(c34-c1345))*u[1, 1+{0}, 1+j, 1+k]**2 + \
                    (tx1*(c34-c1345)+ty1*(r43*c34-c1345)+tz1*(c34-c1345))*u[2, 1+{0}, 1+j, 1+k]**2 + \
                    (tx1*(c34-c1345)+ty1*(c34-c1345)+tz1*(r43*c34-c1345))*u[3, 1+{0}, 1+j, 1+k]**2 \
                )*tmp3 + (tx1+ty1+tz1)*c1345*tmp2*u[4, 1+{0}, 1+j, 1+k])  {{id=s26_write_d_i}}
            d[4, 1, {0}] = dt*2*tmp2*u[1, 1+{0}, 1+j, 1+k]*(tx1*(r43*c34-c1345)+ty1*(c34-c1345)+tz1*(c34-c1345))  {{id=s27_write_d_i}}
            d[4, 2, {0}] = dt*2*tmp2*u[2, 1+{0}, 1+j, 1+k]*(tx1*(c34-c1345)+ty1*(r43*c34-c1345)+tz1*(c34-c1345))  {{id=s28_write_d_i}}
            d[4, 3, {0}] = dt*2*tmp2*u[3, 1+{0}, 1+j, 1+k]*(tx1*(c34-c1345)+ty1*(c34-c1345)+tz1*(r43*c34-c1345))  {{id=s29_write_d_i}}
            d[4, 4, {0}] = 1+dt*2*(tx1+ty1+tz1)*c1345*tmp1+dt*2*(tx1*dx5+ty1*dy5+tz1*dz5)  {{id=s30_write_d_i}}

            tmp1 = rho_i[1+{0}, 1+j, k]  {{id=s31}}
            tmp2 = tmp1*tmp1  {{id=s32}}
            tmp3 = tmp1*tmp2  {{id=s33}}

            <>a[0, 0, {0}] = -dt*tz1*dz1  {{id=s34_write_a_i}}
            a[0, 1, {0}] = 0  {{id=s35_write_a_i}}
            a[0, 2, {0}] = 0  {{id=s36_write_a_i}}
            a[0, 3, {0}] = -dt*tz2  {{id=s37_write_a_i}}
            a[0, 4, {0}] = 0  {{id=s38_write_a_i}}

            a[1, 0, {0}] = -dt*tz2*(-1)*u[1, 1+{0}, 1+j, k]*u[3, 1+{0}, 1+j, k]*tmp2-dt*tz1*(-1)*c34*tmp2*u[1, 1+{0}, 1+j, k]  {{id=s39_write_a_i}}
            a[1, 1, {0}] = -dt*tz2*u[3, 1+{0}, 1+j, k]*tmp1-dt*tz1*c34*tmp1-dt*tz1*dz2  {{id=s40_write_a_i}}
            a[1, 2, {0}] = 0  {{id=s41_write_a_i}}
            a[1, 3, {0}] = -dt*tz2*u[1, 1+{0}, 1+j, k]*tmp1  {{id=s42_write_a_i}}
            a[1, 4, {0}] = 0  {{id=s43_write_a_i}}

            a[2, 0, {0}] = -dt*tz2*(-1)*u[2, 1+{0}, 1+j, k]*u[3, 1+{0}, 1+j, k]*tmp2-dt*tz1*(-1)*c34*tmp2*u[2, 1+{0}, 1+j, k]  {{id=s44_write_a_i}}
            a[2, 1, {0}] = 0  {{id=s45_write_a_i}}
            a[2, 2, {0}] = -dt*tz2*u[3, 1+{0}, 1+j, k]*tmp1-dt*tz1*c34*tmp1-dt*tz1*dz3  {{id=s46_write_a_i}}
            a[2, 3, {0}] = -dt*tz2*u[2, 1+{0}, 1+j, k]*tmp1  {{id=s47_write_a_i}}
            a[2, 4, {0}] = 0  {{id=s48_write_a_i}}

            a[3, 0, {0}] = -dt*tz2*(((-1)*u[3, 1+{0}, 1+j, k]*tmp1)**2+c2*qs[1+{0}, 1+j, k]*tmp1)-dt*tz1*(-1)*r43*c34*tmp2*u[3, 1+{0}, 1+j, k]  {{id=s49_write_a_i}}
            a[3, 1, {0}] = -dt*tz2*(-1)*c2*u[1, 1+{0}, 1+j, k]*tmp1  {{id=s50_write_a_i}}
            a[3, 2, {0}] = -dt*tz2*(-1)*c2*u[2, 1+{0}, 1+j, k]*tmp1  {{id=s51_write_a_i}}
            a[3, 3, {0}] = -dt*tz2*(2-c2)*u[3, 1+{0}, 1+j, k]*tmp1-dt*tz1*r43*c34*tmp1-dt*tz1*dz4  {{id=s52_write_a_i}}
            a[3, 4, {0}] = -dt*tz2*c2  {{id=s53_write_a_i}}

            a[4, 0, {0}] = \
                -dt*tz2*(c2*2*qs[1+{0}, 1+j, k]-c1*u[4, 1+{0}, 1+j, k])*u[3, 1+{0}, 1+j, k]*tmp2 - \
                dt*tz1*((-1)*(c34-c1345)*tmp3*u[1, 1+{0}, 1+j, k]**2 - \
                (c34-c1345)*tmp3*u[2, 1+{0}, 1+j, k]**2 - \
                (r43*c34-c1345)*tmp3*u[3, 1+{0}, 1+j, k]**2 - \
                c1345*tmp2*u[4, 1+{0}, 1+j, k])  {{id=s54_write_a_i}}
            a[4, 1, {0}] = -dt*tz2*(-1)*c2*u[1, 1+{0}, 1+j, k]*u[3, 1+{0}, 1+j, k]*tmp2-dt*tz1*(c34-c1345)*tmp2*u[1, 1+{0}, 1+j, k]  {{id=s55_write_a_i}}
            a[4, 2, {0}] = -dt*tz2*(-1)*c2*u[2, 1+{0}, 1+j, k]*u[3, 1+{0}, 1+j, k]*tmp2-dt*tz1*(c34-c1345)*tmp2*u[2, 1+{0}, 1+j, k]  {{id=s56_write_a_i}}
            a[4, 3, {0}] = \
                -dt*tz2*( \
                    c1*u[4, 1+{0}, 1+j, k]*tmp1 - \
                    c2*(qs[1+{0}, 1+j, k]*tmp1 + u[3, 1+{0}, 1+j, k]*u[3, 1+{0}, 1+j, k]*tmp2) \
                ) - dt*tz1*(r43*c34-c1345)*tmp2*u[3, 1+{0}, 1+j, k]  {{id=s57_write_a_i}}
            a[4, 4, {0}] = -dt*tz2*c1*u[3, 1+{0}, 1+j, k]*tmp1-dt*tz1*c1345*tmp1-dt*tz1*dz5  {{id=s58_write_a_i}}

            tmp1 = rho_i[1+{0}, j, 1+k]  {{id=s59}}
            tmp2 = tmp1*tmp1  {{id=s60}}
            tmp3 = tmp1*tmp2  {{id=s61}}

            <>b[0, 0, {0}] = -dt*ty1*dy1  {{id=s62_write_b_i}}
            b[0, 1, {0}] = 0  {{id=s63_write_b_i}}
            b[0, 2, {0}] = -dt*ty2  {{id=s64_write_b_i}}
            b[0, 3, {0}] = 0  {{id=s65_write_b_i}}
            b[0, 4, {0}] = 0  {{id=s66_write_b_i}}

            b[1, 0, {0}] = -dt*ty2*(-1)*u[1, 1+{0}, j, 1+k]*u[2, 1+{0}, j, 1+k]*tmp2-dt*ty1*(-1)*c34*tmp2*u[1, 1+{0}, j, 1+k]  {{id=s67_write_b_i}}
            b[1, 1, {0}] = -dt*ty2*u[2, 1+{0}, j, 1+k]*tmp1-dt*ty1*c34*tmp1-dt*ty1*dy2  {{id=s68_write_b_i}}
            b[1, 2, {0}] = -dt*ty2*u[1, 1+{0}, j, 1+k]*tmp1  {{id=s69_write_b_i}}
            b[1, 3, {0}] = 0  {{id=s70_write_b_i}}
            b[1, 4, {0}] = 0  {{id=s71_write_b_i}}

            b[2, 0, {0}] = -dt*ty2*(((-1)*u[2, 1+{0}, j, 1+k]*tmp1)**2+c2*qs[1+{0}, j, 1+k]*tmp1)-dt*ty1*(-1)*r43*c34*tmp2*u[2, 1+{0}, j, 1+k]  {{id=s72_write_b_i}}
            b[2, 1, {0}] = -dt*ty2*(-1)*c2*u[1, 1+{0}, j, 1+k]*tmp1  {{id=s73_write_b_i}}
            b[2, 2, {0}] = -dt*ty2*(2-c2)*u[2, 1+{0}, j, 1+k]*tmp1-dt*ty1*r43*c34*tmp1-dt*ty1*dy3  {{id=s74_write_b_i}}
            b[2, 3, {0}] = -dt*ty2*(-1)*c2*u[3, 1+{0}, j, 1+k]*tmp1  {{id=s75_write_b_i}}
            b[2, 4, {0}] = -dt*ty2*c2  {{id=s76_write_b_i}}

            b[3, 0, {0}] = -dt*ty2*(-1)*u[2, 1+{0}, j, 1+k]*u[3, 1+{0}, j, 1+k]*tmp2-dt*ty1*(-1)*c34*tmp2*u[3, 1+{0}, j, 1+k]  {{id=s77_write_b_i}}
            b[3, 1, {0}] = 0  {{id=s78_write_b_i}}
            b[3, 2, {0}] = -dt*ty2*u[3, 1+{0}, j, 1+k]*tmp1  {{id=s79_write_b_i}}
            b[3, 3, {0}] = -dt*ty2*u[2, 1+{0}, j, 1+k]*tmp1-dt*ty1*c34*tmp1-dt*ty1*dy4  {{id=s80_write_b_i}}
            b[3, 4, {0}] = 0  {{id=s81_write_b_i}}

            b[4, 0, {0}] = \
                -dt*ty2*(c2*2*qs[1+{0}, j, 1+k] - c1*u[4, 1+{0}, j, 1+k])*u[2, 1+{0}, j, 1+k]*tmp2 - \
                dt*ty1*( \
                    (-1)*(c34-c1345)*tmp3*u[1, 1+{0}, j, 1+k]**2 - \
                    (r43*c34-c1345)*tmp3*u[2, 1+{0}, j, 1+k]**2 - \
                    (c34-c1345)*tmp3*u[3, 1+{0}, j, 1+k]**2 - \
                    c1345*tmp2*u[4, 1+{0}, j, 1+k])  {{id=s82_write_b_i}}
            b[4, 1, {0}] = \
                -dt*ty2*(-1)*c2*u[1, 1+{0}, j, 1+k]*u[2, 1+{0}, j, 1+k]*tmp2 - \
                dt*ty1*(c34-c1345)*tmp2*u[1, 1+{0}, j, 1+k]  {{id=s83_write_b_i}}
            b[4, 2, {0}] = \
                -dt*ty2*( \
                    c1*u[4, 1+{0}, j, 1+k]*tmp1 - \
                    c2*(qs[1+{0}, j, 1+k]*tmp1 + \
                    u[2, 1+{0}, j, 1+k]*u[2, 1+{0}, j, 1+k]*tmp2)) - \
                    dt*ty1*(r43*c34-c1345)*tmp2*u[2, 1+{0}, j, 1+k]  {{id=s84_write_b_i}}
            b[4, 3, {0}] = \
                -dt*ty2*(-1)*c2*u[2, 1+{0}, j, 1+k]*u[3, 1+{0}, j, 1+k]*tmp2 - \
                dt*ty1*(c34-c1345)*tmp2*u[3, 1+{0}, j, 1+k]  {{id=s85_write_b_i}}
            b[4, 4, {0}] = -dt*ty2*c1*u[2, 1+{0}, j, 1+k]*tmp1-dt*ty1*c1345*tmp1-dt*ty1*dy5  {{id=s86_write_b_i}}

            tmp1 = rho_i[{0}, 1+j, 1+k]  {{id=s87}}
            tmp2 = tmp1*tmp1  {{id=s88}}
            tmp3 = tmp1*tmp2  {{id=s89}}

            <>c[0, 0, {0}] = -dt*tx1*dx1  {{id=s90_write_c_i}}
            c[0, 1, {0}] = -dt*tx2  {{id=s91_write_c_i}}
            c[0, 2, {0}] = 0  {{id=s92_write_c_i}}
            c[0, 3, {0}] = 0  {{id=s93_write_c_i}}
            c[0, 4, {0}] = 0  {{id=s94_write_c_i}}

            c[1, 0, {0}] = \
                -dt*tx2*( \
                    ((-1)*u[1, {0}, 1+j, 1+k]*tmp1)**2 + \
                    c2*qs[{0}, 1+j, 1+k]*tmp1 \
                ) - dt*tx1*(-1)*r43*c34*tmp2*u[1, {0}, 1+j, 1+k]  {{id=s95_write_c_i}}
            c[1, 1, {0}] = -dt*tx2*(2-c2)*u[1, {0}, 1+j, 1+k]*tmp1-dt*tx1*r43*c34*tmp1-dt*tx1*dx2  {{id=s96_write_c_i}}
            c[1, 2, {0}] = -dt*tx2*(-1)*c2*u[2, {0}, 1+j, 1+k]*tmp1  {{id=s97_write_c_i}}
            c[1, 3, {0}] = -dt*tx2*(-1)*c2*u[3, {0}, 1+j, 1+k]*tmp1  {{id=s98_write_c_i}}
            c[1, 4, {0}] = -dt*tx2*c2  {{id=s99_write_c_i}}

            c[2, 0, {0}] = -dt*tx2*(-1)*u[1, {0}, 1+j, 1+k]*u[2, {0}, 1+j, 1+k]*tmp2-dt*tx1*(-1)*c34*tmp2*u[2, {0}, 1+j, 1+k]  {{id=s100_write_c_i}}
            c[2, 1, {0}] = -dt*tx2*u[2, {0}, 1+j, 1+k]*tmp1  {{id=s101_write_c_i}}
            c[2, 2, {0}] = -dt*tx2*u[1, {0}, 1+j, 1+k]*tmp1-dt*tx1*c34*tmp1-dt*tx1*dx3  {{id=s102_write_c_i}}
            c[2, 3, {0}] = 0  {{id=s103_write_c_i}}
            c[2, 4, {0}] = 0  {{id=s104_write_c_i}}

            c[3, 0, {0}] = -dt*tx2*(-1)*u[1, {0}, 1+j, 1+k]*u[3, {0}, 1+j, 1+k]*tmp2-dt*tx1*(-1)*c34*tmp2*u[3, {0}, 1+j, 1+k]  {{id=s105_write_c_i}}
            c[3, 1, {0}] = -dt*tx2*u[3, {0}, 1+j, 1+k]*tmp1  {{id=s106_write_c_i}}
            c[3, 2, {0}] = 0  {{id=s107_write_c_i}}
            c[3, 3, {0}] = -dt*tx2*u[1, {0}, 1+j, 1+k]*tmp1-dt*tx1*c34*tmp1-dt*tx1*dx4  {{id=s108_write_c_i}}
            c[3, 4, {0}] = 0  {{id=s109_write_c_i}}

            c[4, 0, {0}] = \
                -dt*tx2*( \
                    c2*2*qs[{0}, 1+j, 1+k] - c1*u[4, {0}, 1+j, 1+k] \
                )*u[1, {0}, 1+j, 1+k]*tmp2 - \
                dt*tx1*( \
                    (-1)*(r43*c34-c1345)*tmp3*u[1, {0}, 1+j, 1+k]**2 - \
                    (c34-c1345)*tmp3*u[2, {0}, 1+j, 1+k]**2 - \
                    (c34-c1345)*tmp3*u[3, {0}, 1+j, 1+k]**2 - \
                    c1345*tmp2*u[4, {0}, 1+j, 1+k])  {{id=s110_write_c_i}}
            c[4, 1, {0}] = \
                -dt*tx2*( \
                    c1*u[4, {0}, 1+j, 1+k]*tmp1 - \
                    c2*( \
                        u[1, {0}, 1+j, 1+k]*u[1, {0}, 1+j, 1+k]*tmp2 + \
                        qs[{0}, 1+j, 1+k]*tmp1 \
                    ) \
                ) - dt*tx1*(r43*c34-c1345)*tmp2*u[1, {0}, 1+j, 1+k]  {{id=s111_write_c_i}}
            c[4, 2, {0}] = -dt*tx2*(-1)*c2*u[2, {0}, 1+j, 1+k]*u[1, {0}, 1+j, 1+k]*tmp2-dt*tx1*(c34-c1345)*tmp2*u[2, {0}, 1+j, 1+k]  {{id=s112_write_c_i}}
            c[4, 3, {0}] = -dt*tx2*(-1)*c2*u[3, {0}, 1+j, 1+k]*u[1, {0}, 1+j, 1+k]*tmp2-dt*tx1*(c34-c1345)*tmp2*u[3, {0}, 1+j, 1+k]  {{id=s113_write_c_i}}
            c[4, 4, {0}] = -dt*tx2*c1*u[1, {0}, 1+j, 1+k]*tmp1-dt*tx1*c1345*tmp1-dt*tx1*dx5  {{id=s114_write_c_i}}
        end
        """.format(i_iname)  # noqa
    # }}}

    domains = ["[iend, ist] -> {{ [{0}] : ist <= {0} <= iend }}".format(i_iname)]

    return domains, s

# }}}


# {{{ get_blts_knl_str

def get_blts_knl_str(
        i_iname="i", tmp1_name="tmp1",
        ):

    ldx = "c"
    ldy = "b"
    ldz = "a"

    # {{{ knl string
    s = """
        for {0}
            for m
                <>tv[m] = v[m, 1+{0}, 1+j, 1+k] + (-1)*omega*( \
                        {3}[m, 0, {0}]*v[0, 1+{0}, 1+j, k] \
                      + {3}[m, 1, {0}]*v[1, 1+{0}, 1+j, k] \
                      + {3}[m, 2, {0}]*v[2, 1+{0}, 1+j, k] \
                      + {3}[m, 3, {0}]*v[3, 1+{0}, 1+j, k] \
                      + {3}[m, 4, {0}]*v[4, 1+{0}, 1+j, k])  {{id=s0_write_tv_i_dep_a_same_ijk_dep_v_lt_k}}
                tv[m] = tv[m] + (-1)*omega*( \
                        {2}[m, 0, {0}]*v[0, 1+{0}, j, 1+k] \
                      + {1}[m, 0, {0}]*v[0, {0}, 1+j, 1+k] \
                      + {2}[m, 1, {0}]*v[1, 1+{0}, j, 1+k] \
                      + {1}[m, 1, {0}]*v[1, {0}, 1+j, 1+k] \
                      + {2}[m, 2, {0}]*v[2, 1+{0}, j, 1+k] \
                      + {1}[m, 2, {0}]*v[2, {0}, 1+j, 1+k] \
                      + {2}[m, 3, {0}]*v[3, 1+{0}, j, 1+k] \
                      + {1}[m, 3, {0}]*v[3, {0}, 1+j, 1+k] \
                      + {2}[m, 4, {0}]*v[4, 1+{0}, j, 1+k] \
                      + {1}[m, 4, {0}]*v[4, {0}, 1+j, 1+k])  {{id=s1_write_tv_i_dep_b_same_ijk_dep_c_same_ijk_dep_v_lt_ij}}
            end
            for m_1
                <>tmat[m_1, 0] = d[m_1, 0, {0}]  {{id=s2_dep_d_same_ijk}}
                tmat[m_1, 1] = d[m_1, 1, {0}]  {{id=s3_dep_d_same_ijk}}
                tmat[m_1, 2] = d[m_1, 2, {0}]  {{id=s4_dep_d_same_ijk}}
                tmat[m_1, 3] = d[m_1, 3, {0}]  {{id=s5_dep_d_same_ijk}}
                tmat[m_1, 4] = d[m_1, 4, {0}]  {{id=s6_dep_d_same_ijk}}
            end

            <>{4} = 1.0 / tmat[0, 0]  {{id=s7}}

            <>tmp = {4}*tmat[1, 0]  {{id=s8}}
            tmat[1, 1] = tmat[1, 1]+(-1)*tmp*tmat[0, 1]  {{id=s9}}
            tmat[1, 2] = tmat[1, 2]+(-1)*tmp*tmat[0, 2]  {{id=s10}}
            tmat[1, 3] = tmat[1, 3]+(-1)*tmp*tmat[0, 3]  {{id=s11}}
            tmat[1, 4] = tmat[1, 4]+(-1)*tmp*tmat[0, 4]  {{id=s12}}
            tv[1] = tv[1]+(-1)*tv[0]*tmp  {{id=s13_dep_tv_same_ijk}}

            tmp = {4}*tmat[2, 0]  {{id=s14}}
            tmat[2, 1] = tmat[2, 1]+(-1)*tmp*tmat[0, 1]  {{id=s15}}
            tmat[2, 2] = tmat[2, 2]+(-1)*tmp*tmat[0, 2]  {{id=s16}}
            tmat[2, 3] = tmat[2, 3]+(-1)*tmp*tmat[0, 3]  {{id=s17}}
            tmat[2, 4] = tmat[2, 4]+(-1)*tmp*tmat[0, 4]  {{id=s18}}
            tv[2] = tv[2]+(-1)*tv[0]*tmp  {{id=s19_dep_tv_same_ijk}}

            tmp = {4}*tmat[3, 0]  {{id=s20}}
            tmat[3, 1] = tmat[3, 1]+(-1)*tmp*tmat[0, 1]  {{id=s21}}
            tmat[3, 2] = tmat[3, 2]+(-1)*tmp*tmat[0, 2]  {{id=s22}}
            tmat[3, 3] = tmat[3, 3]+(-1)*tmp*tmat[0, 3]  {{id=s23}}
            tmat[3, 4] = tmat[3, 4]+(-1)*tmp*tmat[0, 4]  {{id=s24}}
            tv[3] = tv[3]+(-1)*tv[0]*tmp  {{id=s25_dep_tv_same_ijk}}

            tmp = {4}*tmat[4, 0]  {{id=s26}}
            tmat[4, 1] = tmat[4, 1]+(-1)*tmp*tmat[0, 1]  {{id=s27}}
            tmat[4, 2] = tmat[4, 2]+(-1)*tmp*tmat[0, 2]  {{id=s28}}
            tmat[4, 3] = tmat[4, 3]+(-1)*tmp*tmat[0, 3]  {{id=s29}}
            tmat[4, 4] = tmat[4, 4]+(-1)*tmp*tmat[0, 4]  {{id=s30}}
            tv[4] = tv[4]+(-1)*tv[0]*tmp  {{id=s31_dep_tv_same_ijk}}

            {4} = 1.0 / tmat[1, 1]  {{id=s32}}

            tmp = {4}*tmat[2, 1]  {{id=s33}}
            tmat[2, 2] = tmat[2, 2]+(-1)*tmp*tmat[1, 2]  {{id=s34}}
            tmat[2, 3] = tmat[2, 3]+(-1)*tmp*tmat[1, 3]  {{id=s35}}
            tmat[2, 4] = tmat[2, 4]+(-1)*tmp*tmat[1, 4]  {{id=s36}}
            tv[2] = tv[2]+(-1)*tv[1]*tmp  {{id=s37}}

            tmp = {4}*tmat[3, 1]  {{id=s38}}
            tmat[3, 2] = tmat[3, 2]+(-1)*tmp*tmat[1, 2]  {{id=s39}}
            tmat[3, 3] = tmat[3, 3]+(-1)*tmp*tmat[1, 3]  {{id=s40}}
            tmat[3, 4] = tmat[3, 4]+(-1)*tmp*tmat[1, 4]  {{id=s41}}
            tv[3] = tv[3]+(-1)*tv[1]*tmp  {{id=s42}}

            tmp = {4}*tmat[4, 1]  {{id=s43}}
            tmat[4, 2] = tmat[4, 2]+(-1)*tmp*tmat[1, 2]  {{id=s44}}
            tmat[4, 3] = tmat[4, 3]+(-1)*tmp*tmat[1, 3]  {{id=s45}}
            tmat[4, 4] = tmat[4, 4]+(-1)*tmp*tmat[1, 4]  {{id=s46}}
            tv[4] = tv[4]+(-1)*tv[1]*tmp  {{id=s47}}

            {4} = 1.0 / tmat[2, 2]  {{id=s48}}

            tmp = {4}*tmat[3, 2]  {{id=s49}}
            tmat[3, 3] = tmat[3, 3]+(-1)*tmp*tmat[2, 3]  {{id=s50}}
            tmat[3, 4] = tmat[3, 4]+(-1)*tmp*tmat[2, 4]  {{id=s51}}
            tv[3] = tv[3]+(-1)*tv[2]*tmp  {{id=s52}}

            tmp = {4}*tmat[4, 2]  {{id=s53}}
            tmat[4, 3] = tmat[4, 3]+(-1)*tmp*tmat[2, 3]  {{id=s54}}
            tmat[4, 4] = tmat[4, 4]+(-1)*tmp*tmat[2, 4]  {{id=s55}}
            tv[4] = tv[4]+(-1)*tv[2]*tmp  {{id=s56}}

            {4} = 1.0 / tmat[3, 3]  {{id=s57}}

            tmp = {4}*tmat[4, 3]  {{id=s58}}
            tmat[4, 4] = tmat[4, 4]+(-1)*tmp*tmat[3, 4]  {{id=s59}}
            tv[4] = tv[4]+(-1)*tv[3]*tmp  {{id=s60}}

            v[4, 1+{0}, 1+j, 1+k] = tv[4] / tmat[4, 4]  {{id=s61_write_v_ijk}}

            tv[3] = tv[3] \
                + (-1)*tmat[3, 4]*v[4, 1+{0}, 1+j, 1+k]  {{id=s62_dep_v_s61}}
            v[3, 1+{0}, 1+j, 1+k] = tv[3] / tmat[3, 3]  {{id=s63_write_v_ijk}}

            tv[2] = tv[2] \
                + (-1)*tmat[2, 3]*v[3, 1+{0}, 1+j, 1+k] \
                + (-1)*tmat[2, 4]*v[4, 1+{0}, 1+j, 1+k]  {{id=s64_dep_v_s61_dep_v_s63}}
            v[2, 1+{0}, 1+j, 1+k] = tv[2] / tmat[2, 2]  {{id=s65_write_v_ijk}}

            tv[1] = tv[1] \
                + (-1)*tmat[1, 2]*v[2, 1+{0}, 1+j, 1+k] \
                + (-1)*tmat[1, 3]*v[3, 1+{0}, 1+j, 1+k] \
                + (-1)*tmat[1, 4]*v[4, 1+{0}, 1+j, 1+k]  {{id=s66_dep_v_s61_dep_v_s63_dep_v_s65}}
            v[1, 1+{0}, 1+j, 1+k] = tv[1] / tmat[1, 1]  {{id=s67_write_v_ijk}}

            tv[0] = tv[0] \
                + (-1)*tmat[0, 1]*v[1, 1+{0}, 1+j, 1+k] \
                + (-1)*tmat[0, 2]*v[2, 1+{0}, 1+j, 1+k] \
                + (-1)*tmat[0, 3]*v[3, 1+{0}, 1+j, 1+k] \
                + (-1)*tmat[0, 4]*v[4, 1+{0}, 1+j, 1+k]  {{id=s68_dep_v_s61_dep_v_s63_dep_v_s65_dep_v_s67}}
            v[0, 1+{0}, 1+j, 1+k] = tv[0] / tmat[0, 0]  {{id=s69_write_v_ijk}}
        end
        """.format(i_iname, ldx, ldy, ldz, tmp1_name)  # noqa

    # }}}

    domains = [
        "[iend, ist] -> {{ [m, m_1, {0}] : "
        "0 <= m <= 4 and 0 <= m_1 <= 4 and ist <= {0} <= iend "
        "}}".format(i_iname)]

    return domains, s

# }}}


# {{{ get_insn_ids_from_str

def get_insn_ids_from_str(s):
    import re
    insn_id_indices = [
        string.start()+3 for string in re.finditer('id=', s)]
    insn_ids_ordered = []
    for insn_id_start in insn_id_indices:
        insn_id_end1 = s.find("}", insn_id_start)
        insn_id_end2 = s.find(",", insn_id_start)
        if insn_id_end2 != -1:
            insn_id_end = min(insn_id_end1, insn_id_end2)
        else:
            insn_id_end = insn_id_end1
        insn_ids_ordered.append(s[insn_id_start:insn_id_end])
    return insn_ids_ordered

# }}}


# {{{ add_sequential_deps_to_knl_str

def add_sequential_deps_to_knl_str(s):
    assert "dep=" not in s
    insn_ids_ordered = get_insn_ids_from_str(s)
    new_s = s
    for i in range(len(insn_ids_ordered)-1):
        prev_insn_id = insn_ids_ordered[i]
        insn_id = insn_ids_ordered[i+1]
        new_s = new_s.replace(insn_id, "%s,dep=%s" % (insn_id, prev_insn_id), 1)
    return new_s

# }}}


# {{{ append_to_insn_ids

def append_to_insn_ids(s, suffix):
    assert "dep=" not in s
    insn_ids_ordered = get_insn_ids_from_str(s)

    new_ids = []
    new_s = s
    for insn_id in insn_ids_ordered:
        new_s = new_s.replace(
            "id=%s}" % (insn_id), "id=%s}" % (insn_id+suffix), 1)
        new_ids.append(insn_id+suffix)
    return new_s, new_ids

# }}}


# {{{ items_containing

def items_containing(tgts, strings, expect_success=True):
    found = [s for s in strings if all(tgt in s for tgt in tgts)]
    if expect_success and not found:
        raise ValueError("%s not found." % (tgts))
    return found

# }}}


# {{{ get_loopy_global_args

def get_loopy_global_args(names, shape, dtype, order):
    return [
        lp.GlobalArg(name, dtype=dtype, shape=shape[:], order=order)
        for name in names]


# }}}


# {{{ get_loopy_value_args

def get_loopy_value_args(names, dtype):
    return [lp.ValueArg(name, dtype=dtype) for name in names]


# }}}


# {{{ _isl_map_with_marked_dims

def _isl_map_with_marked_dims(s, placeholder_mark="'"):
    # For creating legible tests, map strings may be created with a placeholder
    # for the 'before' mark. Replace this placeholder with BEFORE_MARK before
    # creating the map.
    # ALSO, if BEFORE_MARK == "'", ISL will ignore this mark when creating
    # variable names, so it must be added manually.
    import islpy as isl
    from loopy.schedule.checker.utils import (
        append_mark_to_isl_map_var_names,
    )
    from loopy.schedule.checker.schedule import (
        BEFORE_MARK
    )
    dt = isl.dim_type
    if BEFORE_MARK == "'":
        # ISL will ignore the apostrophe; manually name the in_ vars
        return append_mark_to_isl_map_var_names(
            isl.Map(s.replace(placeholder_mark, BEFORE_MARK)),
            dt.in_,
            BEFORE_MARK)
    else:
        return isl.Map(s.replace(placeholder_mark, BEFORE_MARK))

# }}}


# {{{ _process_and_linearize

def _process_and_linearize(knl):
    # Return linearization items along with the preprocessed kernel and
    # linearized kernel
    proc_knl = lp.preprocess_kernel(knl)
    lin_knl = lp.get_one_linearized_kernel(proc_knl)
    return lin_knl.linearization, proc_knl, lin_knl

# }}}


# {{{ _align_and_compare_maps

def _align_and_compare_maps(maps):
    from loopy.schedule.checker.utils import (
        ensure_dim_names_match_and_align,
        prettier_map_string,
    )

    for map1, map2 in maps:
        # Align maps and compare
        map1_aligned = ensure_dim_names_match_and_align(map1, map2)
        if map1_aligned != map2:
            print("Maps not equal:")
            print(prettier_map_string(map1_aligned))
            print(prettier_map_string(map2))
        assert map1_aligned == map2

# }}}


# {{{ _compare_dependencies

def _compare_dependencies(knl, deps_expected, return_unsatisfied=False):

    deps_found = {}
    for stmt in knl.instructions:
        if hasattr(stmt, "dependencies") and stmt.dependencies:
            deps_found[stmt.id] = stmt.dependencies

    assert deps_found.keys() == deps_expected.keys()

    for stmt_id_after, dep_dict_found in deps_found.items():

        dep_dict_expected = deps_expected[stmt_id_after]

        # Ensure deps for stmt_id_after match
        assert dep_dict_found.keys() == dep_dict_expected.keys()

        for stmt_id_before, dep_list_found in dep_dict_found.items():

            # Ensure deps from (stmt_id_before -> stmt_id_after) match
            dep_list_expected = dep_dict_expected[stmt_id_before]
            assert len(dep_list_found) == len(dep_list_expected)
            _align_and_compare_maps(zip(dep_list_found, dep_list_expected))

    if not return_unsatisfied:
        return

    # Get unsatisfied deps
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)
    return lp.find_unsatisfied_dependencies(proc_knl, lin_items)


# }}}


# {{{ print_knl_deps

def print_knl_deps(knl):
    for insn in knl.instructions:
        print("%s: %s" % (insn.id, list(insn.dependencies.keys())))

# }}}

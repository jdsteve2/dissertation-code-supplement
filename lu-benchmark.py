import numpy as np
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa
from eval_utils import (
    get_wtime,
    print_stats,
    check_for_nans,
    get_jacld_knl_str,
    get_blts_knl_str,
    get_loopy_global_args,
    get_loopy_value_args,
    add_sequential_deps_to_knl_str,
    append_to_insn_ids,
    _process_and_linearize,
    items_containing,
)
from loopy.schedule.checker.schedule import (
    BEFORE_MARK,
)
from loopy.schedule.checker.utils import (  # noqa
    make_dep_map,
    prettier_map_string,
)
from itertools import product as cartprod


fdtype = np.float64
idtype = np.int32

GATHER_WTIME = True
# GATHER_WTIME = False

CREATE_AND_CHECK_DEPS = True
# CREATE_AND_CHECK_DEPS = False
LINEARIZE_WITH_NEW_DEPS = True
# LINEARIZE_WITH_NEW_DEPS = False

# CHECK_RESULT_VS_UNTRANS = True  # runs out of temporary mem fast
CHECK_RESULT_VS_UNTRANS = False


# {{{ get_input_arrays

def get_input_arrays(queue, isiz1, isiz2, isiz3):
    # from lu_data.f90:
    a = cl.array.empty(queue, (5, 5, isiz1), dtype=fdtype, order="F")  # ldz
    b = cl.array.empty(queue, (5, 5, isiz1), dtype=fdtype, order="F")  # ldy
    c = cl.array.empty(queue, (5, 5, isiz1), dtype=fdtype, order="F")  # ldx
    d = cl.array.empty(queue, (5, 5, isiz1), dtype=fdtype, order="F")

    # assumes ldmx=isiz1, ldmy=ixiz2, ldmz=isiz2
    v = cl.array.empty(  # aka rsd
        queue,
        (5, int(isiz1/2*2+1), int(isiz2/2*2+1), isiz3),
        dtype=fdtype, order="F")
    u = cl.array.empty(
        queue,
        (5, int(isiz1/2*2+1), int(isiz2/2*2+1), isiz3),
        dtype=fdtype, order="F")
    qs = cl.array.empty(
        queue,
        (int(isiz1/2*2+1), int(isiz2/2*2+1), isiz3),
        dtype=fdtype, order="F")
    rho_i = cl.array.empty(
        queue,
        (int(isiz1/2*2+1), int(isiz2/2*2+1), isiz3),
        dtype=fdtype, order="F")

    cl.clrandom.fill_rand(a)
    cl.clrandom.fill_rand(b)
    cl.clrandom.fill_rand(c)
    cl.clrandom.fill_rand(d)
    # to avoid NaNs, subtract 10 from diagonals in d
    for idx3 in range(isiz1):
        for idx2 in range(5):
            for idx1 in range(5):
                if idx1 == idx2:
                    d[idx1, idx2, idx3] += -10.0

    cl.clrandom.fill_rand(v)
    cl.clrandom.fill_rand(u)
    cl.clrandom.fill_rand(qs)
    cl.clrandom.fill_rand(rho_i)

    return a, b, c, d, u, v, qs, rho_i

# }}}


def main():

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # {{{ create knl arg dict

    # {{{ function signature reference

    # how jacld/blts are called in ssor.f90:
    """
         do k = 2, nz -1·

            call sync_left( isiz1, isiz2, isiz3, rsd )
!$omp do schedule(static)
            do j = jst, jend

!---------------------------------------------------------------------
!   form the lower triangular part of the jacobian matrix
!---------------------------------------------------------------------
               call jacld(j, k)
·
!---------------------------------------------------------------------
!   perform the lower triangular solution
!---------------------------------------------------------------------
               call blts( isiz1, isiz2, isiz3,  &
     &                    nx, ny, nz,  &
     &                    omega,  &
     &                    rsd,  &
     &                    a, b, c, d,  &
     &                    ist, iend, j, k )

            end do
!$omp end do nowait
            call sync_right( isiz1, isiz2, isiz3, rsd )

         end do
    """

    # }}}

    # temp = 398  # largest prob size that works in global mem

    # sizes for reported results
    # temp = 256+64+32
    # temp = 256+64
    # temp = 256+32
    temp = 256
    # temp = 256 - 32
    isiz1 = isiz2 = isiz3 = temp  # set in npbparams.h
    nx = ny = nz = temp  # set in inputlu.data.sample  # noqa

    omega = 1.2  # set in inputlu.data.sample
    dt_default = 2.0e0  # from npbparams.h

    # from domain.f90:
    ist = 2
    iend = nx - 1
    jst = 2
    jend = ny - 1
    kst = 2
    kend = nz - 1

    # adjust index numbering to start at 0:
    idx_bump = 2
    ist_recreated = ist-idx_bump
    iend_recreated = iend-idx_bump
    jst_recreated = jst-idx_bump
    jend_recreated = jend-idx_bump
    kst_recreated = kst-idx_bump
    kend_recreated = kend-idx_bump

    (
        a_orig, b_orig, c_orig, d_orig,
        u_orig, v_orig, qs_orig, rho_i_orig,
    ) = get_input_arrays(queue, isiz1, isiz2, isiz3)

    # (create copies because arrays may be reused in another kernel call)
    d = d_orig.copy()
    u = u_orig.copy()
    v = v_orig.copy()
    qs = qs_orig.copy()
    rho_i = rho_i_orig.copy()

    # from lu_data.f90:
    c1 = 1.40e+00
    c2 = 0.40e+00
    c3 = 1.00e-01
    c4 = 1.00e+00
    c5 = 1.40e+00

    # from read_input.f90:
    # ---------------------------------------------------------------------
    #    if input file does not exist, it uses defaults
    #       ipr = 1 for detailed progress output
    #       inorm = how often the norm is printed (once every inorm iterations)
    #       itmax = number of pseudo time steps
    #       dt = time step
    #       omega 1 over-relaxation factor for SSOR
    #       tolrsd = steady state residual tolerance levels
    #       nx, ny, nz = number of grid points in x, y, z directions
    # ---------------------------------------------------------------------
    nx0 = isiz1
    ny0 = isiz2
    nz0 = isiz3
    dt = dt_default

    # from setcoeff.f90
    dxi = 1.0e+00 / (nx0 - 1)
    deta = 1.0e+00 / (ny0 - 1)
    dzeta = 1.0e+00 / (nz0 - 1)
    tx1 = 1.0e+00 / (dxi * dxi)
    tx2 = 1.0e+00 / (2.0e+00 * dxi)
    ty1 = 1.0e+00 / (deta * deta)
    ty2 = 1.0e+00 / (2.0e+00 * deta)
    tz1 = 1.0e+00 / (dzeta * dzeta)
    tz2 = 1.0e+00 / (2.0e+00 * dzeta)
    # diffusion coefficients
    dx1 = dx2 = dx3 = dx4 = dx5 = 0.75e+00
    dy1 = dy2 = dy3 = dy4 = dy5 = 0.75e+00
    dz1 = dz2 = dz3 = dz4 = dz5 = 1.00e+00

    ldmx = isiz1
    ldmy = isiz2
    ldmz = isiz3

    # because a,b,c,d are temps, iend,jend,kend need to be fixed params
    knl_arg_dict_no_abcd = {
        "u": u, "v": v, "qs": qs, "rho_i": rho_i,
        # "iend": iend_recreated, "jend": jend_recreated, "kend": kend_recreated,
        "jend": jend_recreated, "kend": kend_recreated,
        "ldmx": ldmx, "ldmy": ldmy, "ldmz": ldmz,
        }

    # }}}

    # {{{ create jacld + blts and run combined kernel

    # {{{ combine domain and knl strings

    # unique names in jacld/blts
    i_jacld = "i_jacld"
    i_blts = "i_blts"  # rename i in blts
    tmp1_blts = "tmp1_blts"  # rename tmp1 in blts
    assert i_jacld != i_blts

    # {{{ get kernels using helper functions

    _, jacld_knl_str = get_jacld_knl_str(
        i_iname=i_jacld,
        )
    # track parent kernel in statement ids
    jacld_knl_str, jacld_stmt_ids = append_to_insn_ids(
        jacld_knl_str, "_jacld")

    _, blts_knl_str = get_blts_knl_str(
        i_iname=i_blts, tmp1_name=tmp1_blts,
        )
    # track parent kernel in statement ids
    blts_knl_str, blts_stmt_ids = append_to_insn_ids(
        blts_knl_str, "_blts")

    all_stmt_ids = jacld_stmt_ids+blts_stmt_ids

    # }}}

    # {{{ create iname domain constraint strings

    iprime_jacld = i_jacld+BEFORE_MARK
    iprime_blts = i_blts+BEFORE_MARK
    jprime = "j"+BEFORE_MARK
    kprime = "k"+BEFORE_MARK
    mprime = "m"+BEFORE_MARK
    m_1prime = "m_1"+BEFORE_MARK
    dom_str = {
        i_jacld: "ist <= %s <= iend" % (i_jacld),
        iprime_jacld: "ist <= %s <= iend" % (iprime_jacld),
        i_blts: "ist <= %s <= iend" % (i_blts),
        iprime_blts: "ist <= %s <= iend" % (iprime_blts),
        "j": "jst <= j <= jend",
        jprime: "jst <= %s <= jend" % (jprime),
        "k": "kst <= k <= kend",  # kst=2;kend=nz-1
        kprime: "kst <= %s <= kend" % (kprime),
        "m": "0 <= m <= 4",
        mprime: "0 <= %s <= 4" % (mprime),
        "m_1": "0 <= m_1 <= 4",
        m_1prime: "0 <= %s <= 4" % (m_1prime),
        }

    # }}}

    combined_domains = [
        "[iend, ist, jend, jst, kend, kst] -> { [m, m_1, %s, %s, j, k] : "
        "%s and %s and %s and %s and %s and %s "
        "}" % (
            i_jacld, i_blts,
            dom_str["m"], dom_str["m_1"],
            dom_str[i_jacld], dom_str[i_blts],
            dom_str["j"], dom_str["k"]),
        ]

    assumptions = "iend,jend,kend > 0"

    # {{{ kernel str

    combined_knl_str = \
        """
        for k
            for j
                %s
                %s
            end
        end
        """ % (jacld_knl_str, blts_knl_str)

    if not CREATE_AND_CHECK_DEPS or not LINEARIZE_WITH_NEW_DEPS:
        # new deps will not be created, so use old deps to ensure order:
        combined_knl_str = add_sequential_deps_to_knl_str(combined_knl_str)

    # }}}

    # }}}

    # {{{ make kernel

    # {{{ kernel str

    combined_knl = lp.make_kernel(
        combined_domains,
        combined_knl_str,
        get_loopy_global_args(
            names=["u", "v"],
            shape=(5, "ldmx//2*2+1", "ldmy//2*2+1", "ldmz"),
            dtype=fdtype, order="F") +
        get_loopy_global_args(
            names=["qs", "rho_i"],
            shape=("ldmx//2*2+1", "ldmy//2*2+1", "ldmz"),
            dtype=fdtype, order="F") +
        get_loopy_value_args(
            names=["iend", "jend", "kend", "ldmx", "ldmy", "ldmz"],
            dtype=idtype,
            ) +
        get_loopy_value_args(
            names=[
                "dt",
                "tx1", "tx2", "ty1", "ty2", "tz1", "tz2",
                "c1", "c2", "c3", "c4", "c5",
                "dx1", "dx2", "dx3", "dx4", "dx5",
                "dy1", "dy2", "dy3", "dy4", "dy5",
                "dz1", "dz2", "dz3", "dz4", "dz5",
                "omega",
                ],
            dtype=fdtype,
            ),
        seq_dependencies=True,
        assumptions=assumptions,
        silenced_warnings=["insn_count_subgroups_upper_bound"]
        )

    # }}}

    # only do this if v2 deps are created:
    if CREATE_AND_CHECK_DEPS:
        if LINEARIZE_WITH_NEW_DEPS:
            # use new deps instead of old ones
            combined_knl = lp.set_options(combined_knl, use_dependencies_v2=True)
        # (otherwise old deps are used above)

    # }}}

    # {{{ add dtype for temp vars
    combined_knl = lp.add_dtypes(
        combined_knl,
        dict([
            (name, fdtype) for name in
            [
                "r43", "c1345", "c34", "tmp1", "tmp2", "tmp3",
                "tv", tmp1_blts, "tmp", "a", "b", "c", "d",
            ]]))
    # }}}

    if CREATE_AND_CHECK_DEPS:
        print("creating deps...")

        # {{{ Add dependencies

        # {{{ dep creation helper function

        def _ijk_dep(
                constraint_ops, i_iname_depender, i_iname_dependee, self_dep,
                extra_constraints=None, extra_depender_dims=None,
                extra_dependee_dims=None):
            # constraint_ops order: i, j, k

            constraint = "{0}' {2} {1} and j' {3} j and k' {4} k".format(
                i_iname_dependee, i_iname_depender, *constraint_ops)

            # {{{ make domain constraints for relevant i-vars
            i_dom_constraints = []
            if i_iname_depender == i_blts:
                i_dom_constraints.append(dom_str[i_blts])
            else:
                assert i_iname_depender == i_jacld
                i_dom_constraints.append(dom_str[i_jacld])

            if i_iname_dependee == i_blts:
                i_dom_constraints.append(dom_str[iprime_blts])
            else:
                assert i_iname_dependee == i_jacld
                i_dom_constraints.append(dom_str[iprime_jacld])
            assert i_dom_constraints
            i_dom_constraints_str = " and ".join(i_dom_constraints)
            # }}}
            # {{{ create strings for extra constraints/dims if necessary
            if extra_constraints is not None:
                extra_constraints = " and %s" % (extra_constraints)
            else:
                extra_constraints = ""
            if extra_depender_dims is not None:
                extra_depender_dims = ", %s" % (", ".join(extra_depender_dims))
            else:
                extra_depender_dims = ""
            if extra_dependee_dims is not None:
                extra_dependee_dims = ", %s" % (", ".join(extra_dependee_dims))
            else:
                extra_dependee_dims = ""
            # }}}

            return make_dep_map(
                "[ist, iend, jst, jend, kst, kend] -> {{ "
                "[k', j', {0}'{8}] -> [k, j, {1}{9}] : "
                "{2} and {3} and {4} and {5} and {6} "  # domains
                "and {7} {10}"  # constraint
                "}}".format(
                    i_iname_dependee, i_iname_depender,
                    dom_str["k"], dom_str[kprime],
                    dom_str["j"], dom_str[jprime],
                    i_dom_constraints_str,
                    constraint,
                    extra_dependee_dims,
                    extra_depender_dims,
                    extra_constraints
                    ), self_dep=self_dep)

        # }}}

        # {{{ deps from jacld->blts

        # jacld reads u, qs, rho_i (none are written to in jacld or blts, so no deps)
        # jacld writes a, b, c, d (all are read in blts)

        # {{{ a,b,c,d-deps with i=i, j=j, k=k

        # deps from m loop in blts
        sid_pairs = []
        expected_dep_ct = 0
        dependers_a = items_containing(["dep_a_same_ijk"], all_stmt_ids)
        dependees_a = items_containing(["write_a_i"], all_stmt_ids)
        sid_pairs += list(cartprod(
            dependers_a, dependees_a, [("=", "=", "=")],
            [("m",)], [None], [dom_str["m"]]))
        expected_dep_ct += 25
        assert len(sid_pairs) == expected_dep_ct
        dependers_b = items_containing(["dep_b_same_ijk"], all_stmt_ids)
        dependees_b = items_containing(["write_b_i"], all_stmt_ids)
        sid_pairs += list(cartprod(
            dependers_b, dependees_b, [("=", "=", "=")],
            [("m",)], [None], [dom_str["m"]]))
        expected_dep_ct += 25
        assert len(sid_pairs) == expected_dep_ct
        dependers_c = items_containing(["dep_c_same_ijk"], all_stmt_ids)
        dependees_c = items_containing(["write_c_i"], all_stmt_ids)
        sid_pairs += list(cartprod(
            dependers_c, dependees_c, [("=", "=", "=")],
            [("m",)], [None], [dom_str["m"]]))
        expected_dep_ct += 25
        assert len(sid_pairs) == expected_dep_ct
        dependers_d = items_containing(["dep_d_same_ijk"], all_stmt_ids)
        dependees_d = items_containing(["write_d_i"], all_stmt_ids)
        sid_pairs += list(cartprod(
            dependers_d, dependees_d, [("=", "=", "=")],
            [("m_1",)], [None], [dom_str["m_1"]]))
        expected_dep_ct += 125
        assert len(sid_pairs) == expected_dep_ct == 200

        # }}}

        # }}}

        # {{{ deps within blts

        # {{{ global variable deps

        # blts reads a,b,c,d (all are written in blts; deps handled above)
        # blts writes and reads v (aka rsd)

        # deps related to v with i',j',k' + 1 = i,j,k
        dependers_v = items_containing(["dep_v_lt_k"], blts_stmt_ids)
        dependees_v = items_containing(["write_v_ijk"], blts_stmt_ids)
        sid_pairs += list(cartprod(
            dependers_v, dependees_v, [("=", "=", "+ 1 =")],
            [("m",)], [None], [dom_str["m"]]))
        expected_dep_ct += 5
        assert len(sid_pairs) == expected_dep_ct
        dependers_v = items_containing(["dep_v_lt_ij"], blts_stmt_ids)
        dependees_v = items_containing(["write_v_ijk"], blts_stmt_ids)
        sid_pairs += list(cartprod(
            dependers_v, dependees_v, [("+ 1 =", "+ 1 =", "=")],
            [("m",)], [None], [dom_str["m"]]))
        expected_dep_ct += 5
        assert len(sid_pairs) == expected_dep_ct == 210

        # more deps related to v with i',j',k' = i,j,k (at end)
        dependers_v = items_containing(["dep_v_s61"], blts_stmt_ids)
        dependees_v = items_containing(["s61_write_v"], blts_stmt_ids)
        sid_pairs += list(cartprod(
            dependers_v, dependees_v, [("=", "=", "=")],
            [None], [None], [None]))
        expected_dep_ct += 4
        assert len(sid_pairs) == expected_dep_ct == 214

        dependers_v = items_containing(["dep_v_s63"], blts_stmt_ids)
        dependees_v = items_containing(["s63_write_v"], blts_stmt_ids)
        sid_pairs += list(cartprod(
            dependers_v, dependees_v, [("=", "=", "=")],
            [None], [None], [None]))
        expected_dep_ct += 3
        assert len(sid_pairs) == expected_dep_ct == 217

        dependers_v = items_containing(["dep_v_s65"], blts_stmt_ids)
        dependees_v = items_containing(["s65_write_v"], blts_stmt_ids)
        sid_pairs += list(cartprod(
            dependers_v, dependees_v, [("=", "=", "=")],
            [None], [None], [None]))
        expected_dep_ct += 2
        assert len(sid_pairs) == expected_dep_ct == 219

        dependers_v = items_containing(["dep_v_s67"], blts_stmt_ids)
        dependees_v = items_containing(["s67_write_v"], blts_stmt_ids)
        sid_pairs += list(cartprod(
            dependers_v, dependees_v, [("=", "=", "=")],
            [None], [None], [None]))
        expected_dep_ct += 1
        assert len(sid_pairs) == expected_dep_ct == 220

        # }}}

        # {{{ local variable deps
        # (there are a bazillion of these, and they're not all that interesting;
        # create more if time)

        # deps related to tv with i' = i
        # (could be more precise with m... maybe later if time)
        dependers_tv = items_containing(["dep_tv_same_ijk"], blts_stmt_ids)
        dependees_tv = items_containing(["write_tv_i"], blts_stmt_ids)
        sid_pairs += list(cartprod(
            dependers_tv, dependees_tv, [("=", "=", "=")],
            [None], [("m",)], [dom_str["m"]]))
        expected_dep_ct += 4*2
        assert len(sid_pairs) == expected_dep_ct == 228

        # }}}

        # }}}

        # {{{ all sequential deps (some of these are unnecessary, but there are
        # tons of SAME deps here; this is an easy way to ensure we
        # express them all)

        skip_first_x_stmts = 3
        dependers_seq = all_stmt_ids[skip_first_x_stmts+1:]
        dependees_seq = all_stmt_ids[skip_first_x_stmts:-1]

        # {{{ ensure orig still matches new knl
        check_dependers_seq = all_stmt_ids[skip_first_x_stmts+1:]
        check_dependees_seq = all_stmt_ids[skip_first_x_stmts:-1]
        assert dependers_seq == check_dependers_seq
        assert dependees_seq == check_dependees_seq
        # }}}

        for depender, dependee in zip(dependers_seq, dependees_seq):
            depender_insn = combined_knl.id_to_insn[depender]
            dependee_insn = combined_knl.id_to_insn[dependee]
            depender_inames = depender_insn.within_inames
            dependee_inames = dependee_insn.within_inames

            # make sure we don't have any insn within both m and m_1
            # (otherwise logic below needs fixing)
            assert not set(["m", "m_1"]) in depender_inames
            assert not set(["m", "m_1"]) in dependee_inames

            extra_constraints = []
            # add extra dims and constraints
            if "m" in depender_inames:
                extra_depender_dims = ("m",)
                extra_constraints.append(dom_str["m"])
            elif "m_1" in depender_inames:
                extra_depender_dims = ("m_1",)
                extra_constraints.append(dom_str["m_1"])
            else:
                extra_depender_dims = None

            if "m" in dependee_inames:
                extra_dependee_dims = (mprime,)
                extra_constraints.append(dom_str[mprime])
            elif "m_1" in dependee_inames:
                extra_dependee_dims = (m_1prime,)
                extra_constraints.append(dom_str[m_1prime])
            else:
                extra_dependee_dims = None

            # if m or m_1 is in *both* depender and dependee, add SAME constraint
            # (should just update ijk dep func above so that it works for arbitrary
            # inames)
            if "m" in depender_inames & dependee_inames:
                extra_constraints.append("m = %s" % (mprime))
            elif "m_1" in depender_inames & dependee_inames:
                extra_constraints.append("m_1 = %s" % (m_1prime))

            if extra_constraints:
                extra_constraints = " and ".join(extra_constraints)
            else:
                extra_constraints = None

            operators = ("=", "=", "=")

            sid_pairs += [(
                depender, dependee, operators,
                extra_depender_dims, extra_dependee_dims, extra_constraints,
                )]

        expected_dep_ct += len(all_stmt_ids) - 1 - skip_first_x_stmts

        # }}}

        # {{{ add deps to kernel
        ct = 0
        for (
                depender, dependee, operators,
                extra_depender_dims, extra_dependee_dims, extra_constraints
                ) in sid_pairs:
            # {{{ set i inames to i_blts or i_jacld based on sids
            if "blts" in depender:
                i_iname_depender = i_blts
            else:
                assert "jacld" in depender
                i_iname_depender = i_jacld
            if "blts" in dependee:
                i_iname_dependee = i_blts
            else:
                assert "jacld" in dependee
                i_iname_dependee = i_jacld
            # }}}
            d = _ijk_dep(
                operators, i_iname_depender, i_iname_dependee,
                depender == dependee,
                extra_depender_dims=extra_depender_dims,
                extra_dependee_dims=extra_dependee_dims,
                extra_constraints=extra_constraints,
                )
            combined_knl = lp.add_dependency_v2(
                combined_knl, depender, dependee, d)
            ct += 1

        assert ct == expected_dep_ct

        # }}}

        # {{{ verify example dep code snippet shown in dissertation

        knl = combined_knl
        dep_ijk_eq = make_dep_map(
            "[ist, iend, jst, jend, kst, kend] -> { [i_jacld', j', k'] -> [i_blts, j, k, m] : "  # map space # noqa
            "i_blts = i_jacld' and j = j' and k = k' }",  # constraints
            #self_dep=False,  # dependency does not describe a statement depending on itself  # noqa
            knl_with_domains=knl)  # kernel containing domain constraints for indices  # noqa
        dep_k_incr = make_dep_map(
            "[ist, iend, jst, jend, kst, kend] -> { [i_blts', j', k'] -> [i_blts, j, k, m] : "  # map space # noqa
            "i_blts = i_blts' and j = j' and k = k' + 1 }",  # constraints
            #self_dep=False,  # dependency does not describe a statement depending on itself  # noqa
            knl_with_domains=knl)  # kernel containing domain constraints for indices  # noqa

        v_load_stmt_deps = combined_knl.id_to_insn[
            "s0_write_tv_i_dep_a_same_ijk_dep_v_lt_k_blts"].dependencies

        from eval_utils import _align_and_compare_maps
        _align_and_compare_maps([
            (v_load_stmt_deps["s34_write_a_i_jacld"][0], dep_ijk_eq)
            ])
        _align_and_compare_maps([
            (v_load_stmt_deps["s61_write_v_ijk_blts"][0], dep_k_incr)
            ])

        # }}}

        # }}}

    # Fix parameters AFTER deps are added, before saving ref_knl

    # because a,b,c,d are temps, we fix ist,jst,kst, and iend
    combined_knl = lp.fix_parameters(
        combined_knl, ist=ist_recreated, jst=jst_recreated, kst=kst_recreated,
        iend=iend_recreated,
        dt=dt,
        tx1=tx1, tx2=tx2, ty1=ty1, ty2=ty2, tz1=tz1, tz2=tz2,
        c1=c1, c2=c2, c3=c3, c4=c4, c5=c5,
        dx1=dx1, dx2=dx2, dx3=dx3, dx4=dx4, dx5=dx5,
        dy1=dy1, dy2=dy2, dy3=dy3, dy4=dy4, dy5=dy5,
        dz1=dz1, dz2=dz2, dz3=dz3, dz4=dz4, dz5=dz5,
        omega=omega,
        )

    ref_knl = combined_knl

    # diable caching to avoid TypeError for now
    lp.set_caching_enabled(False)

    # {{{ check deps before transformation

    if CREATE_AND_CHECK_DEPS:
        # Get unsatisfied deps
        lin_items, proc_knl, lin_knl = _process_and_linearize(ref_knl)
        unsatisfied_deps = lp.find_unsatisfied_dependencies(proc_knl, lin_items)
        assert not unsatisfied_deps

    # make sure we didn't accidentally change combined_knl
    from loopy.kernel import KernelState
    assert combined_knl.state not in [
        KernelState.PREPROCESSED, KernelState.LINEARIZED]

    # }}}

    print("transforming...")

    # {{{ transform

    # {{{ domain mapping

    # (lines are extra long because this snippet is copied into thesis)
    import islpy as isl
    transform_map = isl.BasicMap(
        "[jend, kend] -> { [j,k] -> [wave, wave_inner] : wave = j + k and wave_inner = j }")  # noqa
    combined_knl = lp.map_domain(combined_knl, transform_map)  # apply index mapping to produce wavefront ordering # noqa

    combined_knl = lp.constrain_loop_nesting(combined_knl, must_not_nest="~wave, wave")  # ensure no loop nests outside 'wave' loop # noqa

    # }}}

    # {{{Check for unsatisfied deps
    if CREATE_AND_CHECK_DEPS:
        lin_items, proc_knl, lin_knl = _process_and_linearize(combined_knl)
        unsatisfied_deps = lp.find_unsatisfied_dependencies(proc_knl, lin_items)
        assert not unsatisfied_deps
    # }}}

    combined_knl = lp.tag_inames(combined_knl, "wave_inner:g.0")  # parallelize diagonal wave fronts across work-groups # noqa

    # {{{ ad wave barrier

    # check sid since using string for thesis snippet
    assert all_stmt_ids[-1] == "s69_write_v_ijk_blts"

    # used in thesis:
    combined_knl = lp.add_barrier(  # add a barrier after the last statement in blts, within the wave loop  # noqa
        combined_knl, within_inames=frozenset(["wave", ]), synchronization_kind="global",  # noqa
        insn_before="id:s69_write_v_ijk_blts", insn_after=None)

    # }}}

    lin_items, proc_knl, lin_knl = _process_and_linearize(combined_knl)
    unsatisfied_deps = lp.find_unsatisfied_dependencies(proc_knl, lin_items)
    assert not unsatisfied_deps

    # Change scope of a,b,c,d arrays so that each thread has its own global array
    combined_knl = lp.privatize_temporaries_with_inames(
        combined_knl, "wave_inner", "a,b,c,d")
    combined_knl = lp.set_temporary_scope(combined_knl, "a,b,c,d", "global")

    # }}}

    # {{{ Check for unsatisfied deps

    if CREATE_AND_CHECK_DEPS:
        lin_items, proc_knl, lin_knl = _process_and_linearize(combined_knl)
        # print(lin_knl.id_to_insn["jacld31"].dependencies)
        unsatisfied_deps = lp.find_unsatisfied_dependencies(proc_knl, lin_items)
        assert not unsatisfied_deps

    # }}}

    print("executing transformed knl...")

    evt, out_combined_knl = combined_knl(queue, **knl_arg_dict_no_abcd)

    print("timing transformed knl...")
    if GATHER_WTIME:
        wtime_combined_knl, _ = get_wtime(
            ctx, queue, combined_knl, knl_arg_dict_no_abcd)

    # }}}

    # {{{ make sure transformed kernel produces same results

    # first check:

    must_nest_seq = ("k", "j", "{%s,%s}" % (i_blts, i_jacld))
    if CHECK_RESULT_VS_UNTRANS:
        # ref_knl: untransformed combined_knl
        ref_knl = lp.constrain_loop_nesting(ref_knl, must_nest=must_nest_seq)

        # double check:
        print("get results from untransformed knl...")
        evt, out_ref_knl = ref_knl(queue, **knl_arg_dict_no_abcd)

        print("comparing...")
        for out_array_combined_knl, out_array_ref_knl in zip(
                out_combined_knl, out_ref_knl):
            array_combined_knl = out_array_combined_knl.get()
            array_ref_knl = out_array_ref_knl.get()
            if not np.allclose(array_combined_knl, array_ref_knl, equal_nan=False):
                print("VALUE MISMATCH")
            else:
                print("Values match!")

    print("get results from global-abcd knl...")

    # }}}

    # {{{ get stats

    if GATHER_WTIME:
        param_dict = {
            "jend": knl_arg_dict_no_abcd["jend"],
            "kend": knl_arg_dict_no_abcd["kend"],
            }
        print("-"*80)
        print("combined kernel stats:")
        (
            flop_rate, throughput, data_moved_bytes, footsize_bytes
        ) = print_stats(combined_knl, param_dict, wtime_combined_knl, fdtype)

    # {{{ nan check

    print("checking for NaNs in combined_knl result...")
    nans_in_combined = False
    for out_array_combined in out_combined_knl:
        if check_for_nans(out_array_combined.get()):
            nans_in_combined = True

    if nans_in_combined:
        print("NANS FOUND!!!!")

    # }}}

    # {{{ save generated cl for combined kernel

    combined_knl_cl = lp.generate_code_v2(combined_knl).device_code()
    from eval_utils import write_to_out_dir
    fname_combined = "combined_knl.cl"
    write_to_out_dir(fname_combined, combined_knl_cl)

    # }}}

    # }}}

    if GATHER_WTIME:
        giga = 10**9
        milli = 10**3
        throughput_ub = data_moved_bytes/wtime_combined_knl
        PEAK_BANDWIDTH_TitanV = 652.8*giga  # noqa
        PEAK_FLOPS64_TitanV = 6144*giga  # noqa
        print("BANDWIDTH UB:")
        table_str = "%d & %3.1f & %3.1f & %3.1f & %3.1f \\\\" % (
            nx,
            wtime_combined_knl*milli,
            flop_rate/giga,
            throughput_ub/giga,
            throughput_ub/PEAK_BANDWIDTH_TitanV*100,
            )
        print(table_str)


if __name__ == "__main__":
    main()

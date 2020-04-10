#include <iostream>
#include <cmath>
#include <cfloat>
#include <assert.h>
#include <limits>

#include <hiopVector.hpp>
#include "testBase.hpp"

namespace hiop::tests {

/**
 * @brief Collection of tests for abstract hiopVector implementations.
 *
 * This class contains implementation of all vector unit tests and abstract
 * interface for testing utility functions, which are specific to vector
 * implementation.
 *
 * @pre All input vectors should be allocated to the same size and have
 * the same partitioning.
 *
 * @post All tests return `true` on all ranks if the test fails on any rank
 * and return `false` otherwise.
 *
 */
class VectorTests : public TestBase
{
public:
    VectorTests(){}
    virtual ~VectorTests(){}

    /*
     * this[i] = 0
     */
    bool vectorSetToZero(hiop::hiopVector& v, int& rank)
    {
        v.setToConstant(one);

        v.setToZero();

        int fail = verifyAnswer(&v, zero);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /// Test get_size() method of hiop vector implementation
    bool vectorGetSize(hiop::hiopVector& x, global_ordinal_type answer, int rank)
    {
        bool fail = (x.get_size() != answer);
        printMessage(fail, __func__, rank);
        return fail;
    }

    /// Test setToConstant method of hiop vector implementation
    bool vectorSetToConstant(hiop::hiopVector& x, int& rank)
    {
        int fail = 0;
        local_ordinal_type N = getLocalSize(&x);

        for(local_ordinal_type i=0; i<N; ++i)
        {
            setLocalElement(&x, i, zero);
        }

        x.setToConstant(one);

        fail = verifyAnswer(&x, one);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &x);
    }

    /*
     * \forall n in n_local if (pattern[n] != 0.0) this[n] = C
     */
    bool vectorSetToConstant_w_patternSelect(
            hiop::hiopVector& x,
            hiop::hiopVector& pattern,
            const int rank)
    {
        const local_ordinal_type N = getLocalSize(&x);
        assert(N == getLocalSize(&pattern));
        static constexpr real_type C = two;
        for (local_ordinal_type i=0; i<N; i++)
        {
            setLocalElement(&x, i, zero);
            setLocalElement(&pattern, i, one);
        }
        if (rank == 0)
            setLocalElement(&pattern, N-1, zero);

        x.setToConstant_w_patternSelect(C, pattern);

        int fail = 0;
        for (local_ordinal_type i=0; i<N; i++)
        {
            const real_type val = getElement(&x, i);
            if (val != C && !(rank == 0 && i == N-1)) fail++;
        }

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &x);
    }

    /*
     * Test for function that copies data from x to this.
     */
    bool vectorCopyFrom(hiop::hiopVector& v, hiop::hiopVector& from, int rank)
    {
        assert(v.get_size() == from.get_size());
        assert(getLocalSize(&v) == getLocalSize(&from));

        from.setToConstant(one);
        v.setToConstant(two);
        v.copyFrom(from);

        int fail = verifyAnswer(&v, one);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    bool vectorCopyFromStarting(
            hiop::hiopVector& x,
            hiop::hiopVector& from,
            int rank)
    {
        int fail = 0;
        const local_ordinal_type N = getLocalSize(&x);
        assert(N == x.get_size() && "This test cannot be ran with distributed vectors");
        assert(N == getLocalSize(&from));
        x.setToConstant(two);

        real_type* _from = (real_type*)malloc(sizeof(real_type) * N);
        for (local_ordinal_type i=0; i<N; i++)
            _from[i] = one;

        if (rank == 0)
        {
            x.copyFromStarting(1, _from, N-1);
        }
        else
        {
            x.copyFromStarting(0, _from, N);
        }

        for (local_ordinal_type i=0; i<N; i++)
        {
            if (getLocalElement(&x, i) != one && !(i == 0 && rank == 0))
                fail++;
        }

        x.setToConstant(two);
        from.setToConstant(one);
        x.copyFromStarting(0, from);
        fail += verifyAnswer(&x, one);

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &x);
    }

    bool vectorStartingAtCopyFromStartingAt(
            hiop::hiopVector& x,
            hiop::hiopVector& from,
            const int rank)
    {
        int fail = 0;
        const local_ordinal_type N = getLocalSize(&x);
        assert(N == x.get_size() && "This test cannot be ran with distributed vectors");
        assert(N == getLocalSize(&from));

        x.setToConstant(one);
        from.setToConstant(two);

        x.startingAtCopyFromStartingAt(1, from, 0);
        for (local_ordinal_type i=0; i<N; i++)
        {
            if (getLocalElement(&x, i) != two && i != 0)
                fail++;
        }

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &x);
    }

    /*
     * Test for function that copies data from this to x.
     */
    bool vectorCopyTo(hiop::hiopVector& v, hiop::hiopVector& to, int rank)
    {
        assert(v.get_size() == to.get_size());
        assert(getLocalSize(&v) == getLocalSize(&to));

        to.setToConstant(one);
        v.setToConstant(two);

        real_type* todata = getLocalData(&to);
        v.copyTo(todata);

        int fail = verifyAnswer(&to, two);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    bool vectorCopyToStarting(
            hiop::hiopVector& x,
            hiop::hiopVector& to,
            const int rank)
    {
        const local_ordinal_type N = getLocalSize(&x);
        assert(N == x.get_size() && "This test cannot be ran with distributed vectors");
        assert(N == getLocalSize(&to));
        int fail = 0;

        x.setToConstant(one);
        to.setToConstant(two);

        x.copyToStarting(to, 0);
        fail += verifyAnswer(&to, one);

        x.setToConstant(one);
        to.setToConstant(two);
        x.copyToStarting(0, to);
        fail += verifyAnswer(&to, one);

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &x);
    }
    
    bool vectorStartingAtCopyToStartingAt(
            hiop::hiopVector& x,
            hiop::hiopVector& to,
            const int rank)
    {
        int fail = 0;
        const local_ordinal_type N = getLocalSize(&x);
        assert(N == x.get_size() && "This test cannot be ran with distributed vectors");
        assert(N == getLocalSize(&to));

        x.setToConstant(one);
        to.setToConstant(two);
        x.startingAtCopyToStartingAt(0, to, 1, N-1);

        for (local_ordinal_type i=0; i<N; i++)
        {
            if (getLocalElement(&to, i) != one && i != 0)
                fail++;
        }

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &x);
    }

    /*
     * this[i] = (pattern[i] == 0 ? 0 : this[i])
     */
    bool vectorSelectPattern(hiop::hiopVector& v, hiop::hiopVector& ix, int rank)
    {
        const local_ordinal_type N = getLocalSize(&v);
        // verify partitioning of test vectors is correct
        assert(v.get_size() == ix.get_size());
        assert(N == getLocalSize(&ix));

        v.setToConstant(two);
        ix.setToConstant(one);
        if (rank== 0)
            setLocalElement(&ix, N - 1, zero);

        v.selectPattern(ix);

        int fail = 0;
        for (local_ordinal_type i=0; i<N; ++i)
        {
            real_type val = getElement(&v, i);
            if ((val != two) && !((rank== 0) && (i == N-1) && (val == zero)))
                fail++;
        }
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * this[i] *= alpha
     */
    bool vectorScale(hiop::hiopVector& v, int rank)
    {
        v.setToConstant(half);
        v.scale(two);

        int fail = verifyAnswer(&v, one);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * this[i] *= x[i]
     */
    bool vectorComponentMult(hiop::hiopVector& v, hiop::hiopVector& x, int& rank)
    {
        assert(v.get_size() == x.get_size());
        assert(getLocalSize(&v) == getLocalSize(&x));

        v.setToConstant(two);
        x.setToConstant(half);

        v.componentMult(x);

        int fail = verifyAnswer(&v, one);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * this[i] /= x[i]
     */
    bool vectorComponentDiv(hiop::hiopVector& v, hiop::hiopVector& x, int rank)
    {
        assert(v.get_size() == x.get_size());
        assert(getLocalSize(&v) == getLocalSize(&x));

        v.setToConstant(one);
        x.setToConstant(two);

        v.componentDiv(x);

        int fail = verifyAnswer(&v, half);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * this[i] = (pattern[i] == 0 ? 0 : this[i]/x[i])
     */
    bool vectorComponentDiv_p_selectPattern(
            hiop::hiopVector& v,
            hiop::hiopVector& x,
            hiop::hiopVector& pattern,
            int rank)
    {
        const local_ordinal_type N = getLocalSize(&v);
        assert(v.get_size() == x.get_size());
        assert(v.get_size() == pattern.get_size());
        assert(N == getLocalSize(&x));
        assert(N == getLocalSize(&pattern));

        v.setToConstant(one);
        x.setToConstant(two);
        pattern.setToConstant(one);
        if (rank== 0)
            setLocalElement(&v, N - 1, zero);

        v.componentDiv_p_selectPattern(x, pattern);

        int fail = 0;
        for (local_ordinal_type i=0; i<N; ++i)
        {
            real_type val = getElement(&v, i);
            if ((val != half) && !((rank== 0) && (i == N-1) && (val == zero)))
                fail++;
        }
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * Test computing 1-norm ||v||  of vector v
     *                            1
     */
    bool vectorOnenorm(hiop::hiopVector& v, int rank)
    {
        v.setToConstant(-one);
        real_type actual = v.onenorm();
        real_type expected = static_cast<real_type>(v.get_size());

        int fail = (actual != expected);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * Test computing 2-norm ||v||  of vector v
     *                            2
     */
    bool vectorTwonorm(hiop::hiopVector& v, int rank)
    {
        v.setToConstant(-one);
        real_type actual = v.twonorm();
        const real_type expected = sqrt(static_cast<real_type>(v.get_size()));

        int fail = !isEqual(expected, actual);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * Test infinity-norm = max(abs(this[i]))
     *                       i
     */
    bool vectorInfnorm(hiop::hiopVector& v, int rank)
    {
        const local_ordinal_type N = getLocalSize(&v);
        const real_type expected = two;

        v.setToConstant(one);
        if (rank== 0)
            setElement(&v, N-1, -two);
        real_type actual = v.infnorm();

        int fail = (expected != actual);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * this[i] += alpha * x[i]
     */
    bool vectorAxpy(hiop::hiopVector& v, hiop::hiopVector& x, int rank)
    {
        const local_ordinal_type N = getLocalSize(&v);
        assert(v.get_size() == x.get_size());
        assert(N == getLocalSize(&x));

        const real_type alpha = half;
        x.setToConstant(two);
        v.setToConstant(one);

        v.axpy(alpha, x);

        int fail = verifyAnswer(&v, two);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * this[i] += alpha * x[i] * z[i]
     */
    bool vectorAxzpy(
            hiop::hiopVector& v,
            hiop::hiopVector& x,
            hiop::hiopVector& z,
            int rank)
    {
        const local_ordinal_type N = getLocalSize(&v);
        assert(v.get_size() == x.get_size());
        assert(N == getLocalSize(&x));

        const real_type alpha = half;
        x.setToConstant(two);
        z.setToConstant(-one);
        v.setToConstant(one);

        v.axzpy(alpha, x, z);

        int fail = verifyAnswer(&v, zero);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * this[i] += alpha * x[i] / z[i]
     */
    bool vectorAxdzpy(
            hiop::hiopVector& v,
            hiop::hiopVector& x,
            hiop::hiopVector& z,
            int rank)
    {
        const local_ordinal_type N = getLocalSize(&v);
        assert(v.get_size() == x.get_size());
        assert(N == getLocalSize(&x));

        const int alpha = two;
        x.setToConstant(-one);
        z.setToConstant(half);
        v.setToConstant(two);

        v.axdzpy(alpha, x, z);

        int fail = verifyAnswer(&v, -two);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * this += C
     */
    bool vectorAddConstant(hiop::hiopVector& x, int rank)
    {
        int fail = 0;

        x.setToConstant(zero);
        x.addConstant(two);

        fail = verifyAnswer(&x, two);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &x);
    }

    /*
     * if (pattern[i] > 0.0) this[i] += C
     */
    bool vectorAddConstant_w_patternSelect(
            hiop::hiopVector& x, 
            hiop::hiopVector& pattern,
            int rank)
    {
        const local_ordinal_type N = getLocalSize(&x);
        assert(pattern.get_size() == x.get_size());
        assert(N == getLocalSize(&pattern));

        x.setToConstant(zero);
        x.addConstant(half);

        if (rank== 0)
            setLocalElement(&x, N - 1, zero);

        int fail = 0;
        for (local_ordinal_type i=0; i<N; ++i)
        {
            real_type val = getElement(&x, i);
            if ((val != half) && !((rank==0) && (i == N-1) && (val == zero)))
                fail++;
        }

        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &x);
    }

    /*
     * Dot product == \sum{this[i] * other[i]}
     */
    bool vectorDotProductWith(
            hiop::hiopVector& x,
            hiop::hiopVector& y,
            const int rank)
    {
        // Must use global size, as every rank will get global
        const global_ordinal_type N = x.get_size(); 
        assert(getLocalSize(&x) == getLocalSize(&y));

        x.setToConstant(one);
        y.setToConstant(two);

        const real_type expected = two * (real_type)N;
        const real_type actual = x.dotProductWith(y);
        const bool fail = !isEqual(actual, expected);

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &x);
    }

    /* 
     * this[i] == -this_prev[i]
     */
    bool vectorNegate(hiop::hiopVector& x, int rank)
    {
        x.setToConstant(one);
        x.negate();
        const bool fail = verifyAnswer(&x, -one);
        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &x);
    }

    bool vectorInvert(hiop::hiopVector& x, int rank)
    {
        x.setToConstant(two);
        x.invert();
        const bool fail = verifyAnswer(&x, half);
        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &x);
    }

    /* 
     * sum{ln(x_i):i=1,..,n}
     */
    bool vectorLogBarrier(
            hiop::hiopVector& x,
            hiop::hiopVector& select,
            int rank)
    {
        const local_ordinal_type N = getLocalSize(&x);
        assert(N == getLocalSize(&select));
        select.setToConstant(one);
        x.setToConstant(two);

        setLocalElement(&select, N-1, 0.0);

        real_type expected = 0.0;
        for (local_ordinal_type i=0; i<N-1; ++i) expected += log(two);
        const real_type res = x.logBarrier(select);

        const bool fail = !isEqual(res, expected);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &x);
    }

    /*
     * this += alpha / select(x)
     */
    bool vectorAddLogBarrierGrad(
            hiop::hiopVector& x,
            hiop::hiopVector& y,
            hiop::hiopVector& select,
            const int rank)
    {
        const local_ordinal_type N = getLocalSize(&x);
        assert(N == getLocalSize(&select));
        assert(N == getLocalSize(&y));
        static constexpr real_type alpha = half;

        select.setToConstant(one);
        x.setToConstant(one);
        y.setToConstant(two);

        if (rank == 0)
            setLocalElement(&select, N-1, 0.0);

        static constexpr real_type expected = one + (alpha / two);
        x.addLogBarrierGrad(alpha, y, select);

        int fail = 0;
        for (local_ordinal_type i=0; i<N; ++i)
        {
            real_type val = getElement(&x, i);
            if ((val != expected) && !((rank==0) && (i == N-1) && (val == one)))
                fail++;
        }

        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &x);
    }

    /*
     * term := 0.0
     * \forall n \in n_local
     *     if left[n] == 1.0 \land right[n] == 0.0
     *         term += this[n]
     * term *= mu * kappa
     * return term
     */
    bool vectorLinearDampingTerm(
            hiop::hiopVector& x,
            hiop::hiopVector& left,
            hiop::hiopVector& right,
            const int rank)
    {
        const local_ordinal_type N = getLocalSize(&x);
        assert(N == getLocalSize(&left));
        assert(N == getLocalSize(&right));
        static constexpr real_type mu = two;
        static constexpr real_type kappa_d = two;

        x.setToConstant(one);
        left.setToConstant(one);
        right.setToConstant(zero);

        if (rank == 0)
        {
            setLocalElement(&left, N-1, two);
            setLocalElement(&right, N-1, two);
        }

        real_type expected = 0.0;
        for (local_ordinal_type i=0; i<N-1; ++i)
        {
            expected += one;
        }
        if (rank != 0) expected += one;
        expected *= mu;
        expected *= kappa_d;

        const real_type term = x.linearDampingTerm(left, right, mu, kappa_d);

        const int fail = !isEqual(term, expected);

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &x);
    }

    /*
     * this[i] > 0
     */
    bool vectorAllPositive(hiop::hiopVector& x, const int rank)
    {
        const local_ordinal_type N = getLocalSize(&x);
        int fail = 0;
        x.setToConstant(one);
        if (!x.allPositive())
            fail++;

        x.setToConstant(one);
        if (rank == 0)
            setLocalElement(&x, N-1, -one);
        if (x.allPositive())
            fail++;

        printMessage(fail, __func__, rank);
        return fail;
    }

    /*
     * this[i] > 0 \lor pattern[i] != 1.0
     */
    bool vectorAllPositive_w_patternSelect(
            hiop::hiopVector& x,
            hiop::hiopVector& pattern,
            const int rank)
    {
        const local_ordinal_type N = getLocalSize(&x);
        assert(N == getLocalSize(&pattern));

        int fail = 0;

        x.setToConstant(one);
        pattern.setToConstant(one);
        if (!x.allPositive_w_patternSelect(pattern))
            fail++;

        x.setToConstant(-one);
        if (x.allPositive_w_patternSelect(pattern))
            fail++;

        x.setToConstant(one);
        if (rank == 0)
            setLocalElement(&x, N-1, -one);
        if (x.allPositive_w_patternSelect(pattern))
            fail++;

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &x);
    }

    /*
     * This method is not yet implemented in HIOP
    bool vectorMin(const hiop::hiopVector& x, const int rank)
    {
        (void)x;
        printMessage(SKIP_TEST, __func__, rank);
        return 0;
    }
    */

    /*
     * Project vector into bounds
     */
    bool vectorProjectIntoBounds(
            hiop::hiopVector& x,
            hiop::hiopVector& lower,
            hiop::hiopVector& upper,
            hiop::hiopVector& lower_pattern,
            hiop::hiopVector& upper_pattern,
            const int rank)
    {
        // setup constants and make assertions
        const local_ordinal_type N = getLocalSize(&x);
        assert(N == getLocalSize(&lower));
        assert(N == getLocalSize(&upper));
        assert(N == getLocalSize(&lower_pattern));
        assert(N == getLocalSize(&upper_pattern));
        static constexpr real_type kappa1 = half;
        static constexpr real_type kappa2 = half;
        int fail = 0;

        // Check that lower > upper returns false
        x.setToConstant(one);
        lower.setToConstant(one);
        upper.setToConstant(-one);
        lower_pattern.setToConstant(one);
        upper_pattern.setToConstant(one);
        if (x.projectIntoBounds(
                    lower, lower_pattern,
                    upper, upper_pattern,
                    kappa1, kappa2))
            fail++;

        // check that patterns are correctly applied and
        // x[0] is left at 1
        x.setToConstant(one);
        lower.setToConstant(-one);
        upper.setToConstant(one);
        lower_pattern.setToConstant(one);
        setLocalElement(&lower_pattern, 0, zero);
        upper_pattern.setToConstant(one);
        setLocalElement(&upper_pattern, 0, zero);

        // Call should return true
        fail += !x.projectIntoBounds(
                lower, lower_pattern, upper,
                upper_pattern, kappa1, kappa2);

        // First element should be one
        fail += !isEqual(getLocalElement(&x, 0), one);

        // Testing when x is on a boundary:
        // Check that projection of 1 into (-1, 1)
        // returns `true' and x == half
        x.setToConstant(one);
        lower.setToConstant(-one);
        upper.setToConstant(one);
        lower_pattern.setToConstant(one);
        upper_pattern.setToConstant(one);
        x.projectIntoBounds(
                lower, lower_pattern, upper,
                upper_pattern, kappa1, kappa2);

        // x[i] == 1/2 \forall i \in [1, N)
        fail += verifyAnswer(&x, half);

        // testing when x is below boundaries
        // check that projection of -2 into (0, 2)
        // returns `true' and x == half
        x.setToConstant(-two);
        lower.setToConstant(zero);
        upper.setToConstant(two);
        lower_pattern.setToConstant(one);
        upper_pattern.setToConstant(one);

        // Call should return true
        fail += !x.projectIntoBounds(
                lower, lower_pattern, upper,
                upper_pattern, kappa1, kappa2);

        // x[i] == 1/2 \forall i \in [1, N)
        fail += verifyAnswer(&x, half);

        // testing when x is above boundaries
        // check that projection of -2 into (0, 2)
        // returns `true' and x == half
        x.setToConstant(two);
        lower.setToConstant(-two);
        upper.setToConstant(zero);
        lower_pattern.setToConstant(one);
        upper_pattern.setToConstant(one);

        // Call should return true
        fail += !x.projectIntoBounds(
                lower, lower_pattern, upper,
                upper_pattern, kappa1, kappa2);

        // x[i] == -1/2 \forall i \in [1, N)
        fail += verifyAnswer(&x, -half);

        printMessage(fail, __func__, rank);
        return 0;
    }

    /*
     * fractionToTheBdry psuedocode:
     *
     * \forall dxi \in dx, dxi >= 0 \implies
     *     return 1.0
     *
     * \exists dxi \in dx s.t. dxi < 0 \implies
     *     return_value := 1.0
     *     auxilary := 0.0
     *     \forall n \in n_local
     *         auxilary = compute_step_to_boundary(x[n], dx[n])
     *         if auxilary < return_value
     *             return_value = auxilary
     *     return auxilary
     */
    bool vectorFractionToTheBdry(
            hiop::hiopVector& x,
            hiop::hiopVector& dx,
            const int rank)
    {
        const local_ordinal_type N = getLocalSize(&x);
        assert(N == getLocalSize(&dx));
        static constexpr real_type tau = half;
        int fail = 0;

        x.setToConstant(one);

        dx.setToConstant(one);
        real_type result = x.fractionToTheBdry(dx, tau);

        real_type expected = one;
        fail += !isEqual(result, expected);

        dx.setToConstant(-one);
        result = x.fractionToTheBdry(dx, tau);
        real_type aux;
        expected = one;
        for (local_ordinal_type i=0; i<N; i++)
        {
            aux = -tau * getLocalElement(&x, i) / getLocalElement(&dx, i);
            if (aux<expected) expected=aux;
        }
        fail += !isEqual(result, expected);

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &x);
    }

    /*
     * Same as fractionToTheBdry, except that
     * no x[i] where pattern[i]==0 will be calculated
     */
    bool vectorFractionToTheBdry_w_pattern(
            hiop::hiopVector& x,
            hiop::hiopVector& dx,
            hiop::hiopVector& pattern,
            const int rank)
    {
        const local_ordinal_type N = getLocalSize(&x);
        assert(N == getLocalSize(&dx));
        assert(N == getLocalSize(&pattern));
        static constexpr real_type tau = half;
        int fail = 0;

        // Fraction to boundary is const, so no need to reset x after each test
        x.setToConstant(one);

        // Pattern all ones, X all ones, result should be
        // default (alpha == one)
        pattern.setToConstant(one);
        dx.setToConstant(one);
        real_type result = x.fractionToTheBdry_w_pattern(dx, tau, pattern);
        real_type expected = one;  // default value if dx >= 0
        fail += !isEqual(result, expected);

        // Pattern all ones except for one value, should still be default
        // value of one
        pattern.setToConstant(one);
        if (rank == 0)
            setLocalElement(&pattern, N-1, 0);
        dx.setToConstant(one);
        result = x.fractionToTheBdry_w_pattern(dx, tau, pattern);
        expected = one;  // default value if dx >= 0
        fail += !isEqual(result, expected);

        // Pattern all ones, dx will be <0
        pattern.setToConstant(one);
        dx.setToConstant(-one);
        result = x.fractionToTheBdry_w_pattern(dx, tau, pattern);
        real_type aux;
        expected = one;
        for (int i=0; i<N; i++)
        {
            if (rank == 0 && i == N-1) continue;
            aux = -tau * getLocalElement(&x, i) / getLocalElement(&dx, i);
            if (aux<expected) expected=aux;
        }
        fail += !isEqual(result, expected);

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &x);
    }

    /*
     *  pattern != 0 \lor this == 0
     */
    bool vectorMatchesPattern(
            hiop::hiopVector& x,
            hiop::hiopVector& pattern,
            const int rank)
    {
        const int N = getLocalSize(&x);
        assert(N == getLocalSize(&pattern));
        int fail = 0;

        x.setToConstant(one);
        pattern.setToConstant(one);
        if (!x.matchesPattern(pattern)) fail++;

        x.setToConstant(one);
        pattern.setToConstant(one);
        if (rank == 0) setLocalElement(&pattern, N-1, 0);
        if (x.matchesPattern(pattern)) fail++;

        x.setToConstant(one);
        pattern.setToConstant(one);
        if (rank == 0) setLocalElement(&x, N-1, 0);
        if (!x.matchesPattern(pattern)) fail++;

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &x);
    }

    /*
     * Checks that hiop correctly adjusts based on the
     * hessian of the duals function
     */
    bool vectorAdjustDuals_plh(
            hiop::hiopVector& z1,
            hiop::hiopVector& z2,
            hiop::hiopVector& x,
            hiop::hiopVector& pattern,
            const int rank)
    {
        const int N = getLocalSize(&z1);
        assert(N == getLocalSize(&z2));
        assert(N == getLocalSize(&x));
        assert(N == getLocalSize(&pattern));

        // z1 will adjust duals with it's method
        z1.setToConstant(one);

        // z2's duals will be adjusted by hand
        z2.setToConstant(one);

        x.setToConstant(two);
        pattern.setToConstant(one);

        static constexpr real_type mu = half;
        static constexpr real_type kappa = half;
        z1.adjustDuals_plh(
                x,
                pattern,
                mu,
                kappa);

        real_type a, b;
        for (int i=0; i<N; i++)
        {
            a = mu / getLocalElement(&x, i);
            b = a / kappa;
            a *= kappa;
            if      (getLocalElement(&x, i) < b)     setLocalElement(&z2, i, b);
            else if (a <= b)                    setLocalElement(&z2, i, b);
            else if (a < getLocalElement(&x, i))     setLocalElement(&z2, i, a);
        }

        // the method's adjustDuals_plh should yield
        // the same result as computing by hand
        int fail = 0;
        for (int i=0; i<N; i++)
        {
            fail += !isEqual(
                    getLocalElement(&z1, i),     // expected
                    getLocalElement(&z2, i));    // actual
        }

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &x);
    }

    /*
     * \exists e \in this s.t. isnan(e)
     */
    bool vectorIsnan(hiop::hiopVector& x, const int rank)
    {
        const int N = getLocalSize(&x);
        int fail = 0;
        x.setToConstant(zero);
        if (x.isnan())
            fail++;

        if (rank == 0)
            setLocalElement(&x, N-1, NAN);
        if (x.isnan() && rank != 0)
            fail++;

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &x);
    }

    /*
     * \exists e \in this s.t. isinf(e)
     */
    bool vectorIsinf(hiop::hiopVector& x, const int rank)
    {
        const int N = getLocalSize(&x);
        int fail = 0;
        x.setToConstant(zero);
        if (x.isinf())
            fail++;

        if (rank == 0)
            setLocalElement(&x, N-1, INFINITY);
        if (x.isinf() && rank != 0)
            fail++;

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &x);
    }

    /*
     * \forall e \in this, isfinite(e)
     */
    bool vectorIsfinite(hiop::hiopVector& x, const int rank)
    {
        const local_ordinal_type N = getLocalSize(&x);
        int fail = 0;
        x.setToConstant(zero);
        if (!x.isfinite())
            fail++;

        if (rank == 0)
            setLocalElement(&x, N-1, INFINITY);
        if (!x.isfinite() && rank != 0)
            fail++;

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &x);
    }

protected:
    // Interface to methods specific to vector implementation
    virtual void setElement(hiop::hiopVector* x, int i, real_type val) = 0;
    virtual real_type getElement(const hiop::hiopVector* x, int i) = 0;
    virtual local_ordinal_type getLocalSize(const hiop::hiopVector* x) = 0;
    virtual real_type* getLocalData(hiop::hiopVector* x) = 0;
    virtual int verifyAnswer(hiop::hiopVector* x, real_type answer) = 0;
    virtual bool reduceReturn(int failures, hiop::hiopVector* x) = 0;
};

} // namespace hiop::tests

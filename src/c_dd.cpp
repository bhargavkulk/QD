/*
 * src/c_dd.cc
 *
 * This work was supported by the Director, Office of Science, Division
 * of Mathematical, Information, and Computational Sciences of the
 * U.S. Department of Energy under contract number DE-AC03-76SF00098.
 *
 * Copyright (c) 2000-2001
 *
 * Contains the C wrapper functions for double-double precision arithmetic.
 * This can be used from Fortran code.
 */
#include <cstring>
#include <cstdint>
#include <cmath>

#include "config.h"
#include <qd/dd_real.h>
#include <qd/c_dd.h>

#define TO_DOUBLE_PTR(a, ptr) ptr[0] = a.x[0]; ptr[1] = a.x[1];

#if INT_MAX > 32767
#  define IC(x) ((int32_t) x)
#  define UC(x) ((uint32_t) x)
#else
#  define IC(x) ((int32_t) x##L)
#  define UC(x) ((uint32_t) x##UL)
#endif

#ifndef __BYTE_ORDER__
#  error "endianness not defined"
#endif

#ifndef __FLOAT_WORD_ORDER__
#define __FLOAT_WORD_ORDER__ __BYTE_ORDER__
#endif

#if __FLOAT_WORD_ORDER__ == __ORDER_BIG_ENDIAN__

typedef union
{
  double value;
  struct
  {
    uint32_t msw;
    uint32_t lsw;
  } parts;
} ieee_double_shape_type;

#endif

#if __FLOAT_WORD_ORDER__ == __ORDER_LITTLE_ENDIAN__

typedef union
{
  double value;
  struct
  {
    uint32_t lsw;
    uint32_t msw;
  } parts;
} ieee_double_shape_type;

#endif

#define IEEE754_DOUBLE_MAXEXP	0x7ff
#define IEEE754_DOUBLE_BIAS	0x3ff /* Added to exponent.  */
#define IEEE754_DOUBLE_SHIFT    20

/* Get the more significant 32 bit int from a double.  */

#define GET_HIGH_WORD(i,d)					\
{								\
  const ieee_double_shape_type *gh_u = (const ieee_double_shape_type *)&(d);					\
  (i) = gh_u->parts.msw;						\
}

/* Get the less significant 32 bit int from a double.  */

#define GET_LOW_WORD(i,d)					\
{								\
  const ieee_double_shape_type *gl_u = (const ieee_double_shape_type *)&(d);					\
  (i) = gl_u->parts.lsw;						\
}

/* Set a double from two 32 bit ints.  */

#define INSERT_WORDS(d,ix0,ix1)                                 \
do {                                                            \
  ieee_double_shape_type *iw_u = (ieee_double_shape_type *)&(d);\
  iw_u->parts.msw = (ix0);                                      \
  iw_u->parts.lsw = (ix1);                                      \
} while (0)

int32_t __kernel_rem_pio2(double *x, double *y, int32_t e0, int32_t nx, int prec)
{
	int32_t jz, jx, jv, jp, jk, carry, n, iq[20], i, j, k, m, q0, ih;
	double z, fw, f[20], fq[20], q[20];

	static const int init_jk[] = { 2, 3, 4, 6 };	/* initial value for jk */

	/*
	 * Constants:
	 * The hexadecimal values are the intended ones for the following
	 * constants. The decimal values may be used, provided that the
	 * compiler will convert from decimal to binary accurately enough
	 * to produce the hexadecimal values shown.
	 */
	
	static const double PIo2[] = {
		1.57079625129699707031e+00,			/* 0x3FF921FB, 0x40000000 */
		7.54978941586159635335e-08,			/* 0x3E74442D, 0x00000000 */
		5.39030252995776476554e-15,			/* 0x3CF84698, 0x80000000 */
		3.28200341580791294123e-22,			/* 0x3B78CC51, 0x60000000 */
		1.27065575308067607349e-29,			/* 0x39F01B83, 0x80000000 */
		1.22933308981111328932e-36,			/* 0x387A2520, 0x40000000 */
		2.73370053816464559624e-44,			/* 0x36E38222, 0x80000000 */
		2.16741683877804819444e-51			/* 0x3569F31D, 0x00000000 */
	};

	/*
	 * Table of constants for 2/pi, 396 Hex digits (476 decimal) of 2/pi
	 */
	static const int32_t two_over_pi[] = {
		IC(0xA2F983), IC(0x6E4E44), IC(0x1529FC), IC(0x2757D1), IC(0xF534DD), IC(0xC0DB62),
		IC(0x95993C), IC(0x439041), IC(0xFE5163), IC(0xABDEBB), IC(0xC561B7), IC(0x246E3A),
		IC(0x424DD2), IC(0xE00649), IC(0x2EEA09), IC(0xD1921C), IC(0xFE1DEB), IC(0x1CB129),
		IC(0xA73EE8), IC(0x8235F5), IC(0x2EBB44), IC(0x84E99C), IC(0x7026B4), IC(0x5F7E41),
		IC(0x3991D6), IC(0x398353), IC(0x39F49C), IC(0x845F8B), IC(0xBDF928), IC(0x3B1FF8),
		IC(0x97FFDE), IC(0x05980F), IC(0xEF2F11), IC(0x8B5A0A), IC(0x6D1F6D), IC(0x367ECF),
		IC(0x27CB09), IC(0xB74F46), IC(0x3F669E), IC(0x5FEA2D), IC(0x7527BA), IC(0xC7EBE5),
		IC(0xF17B3D), IC(0x0739F7), IC(0x8A5292), IC(0xEA6BFB), IC(0x5FB11F), IC(0x8D5D08),
		IC(0x560330), IC(0x46FC7B), IC(0x6BABF0), IC(0xCFBC20), IC(0x9AF436), IC(0x1DA9E3),
		IC(0x91615E), IC(0xE61B08), IC(0x659985), IC(0x5F14A0), IC(0x68408D), IC(0xFFD880),
		IC(0x4D7327), IC(0x310606), IC(0x1556CA), IC(0x73A8C9), IC(0x60E27B), IC(0xC08C6B)
	};

	static const double zero = 0.0;
	static const double one = 1.0;
	static const double two24 = 1.67772160000000000000e+07;		/* 0x41700000, 0x00000000 */
	static const double twon24 = 5.96046447753906250000e-08;	/* 0x3E700000, 0x00000000 */

	/* initialize jk */
	jk = init_jk[prec];
	jp = jk;

	/* determine jx,jv,q0, note that 3>q0 */
	jx = nx - 1;
	jv = (e0 - 3) / 24;
	if (jv < 0)
		jv = 0;
	q0 = e0 - 24 * (jv + 1);

	/* set up f[0] to f[jx+jk] where f[jx+jk] = two_over_pi[jv+jk] */
	j = jv - jx;
	m = jx + jk;
	for (i = 0; i <= m; i++, j++)
		f[i] = (j < 0) ? zero : (double) two_over_pi[j];

	/* compute q[0],q[1],...q[jk] */
	for (i = 0; i <= jk; i++)
	{
		for (j = 0, fw = 0.0; j <= jx; j++)
			fw += x[j] * f[jx + i - j];
		q[i] = fw;
	}

	jz = jk;
  recompute:
	/* distill q[] into iq[] reversingly */
	for (i = 0, j = jz, z = q[jz]; j > 0; i++, j--)
	{
		fw = (double) ((int32_t) (twon24 * z));
		iq[i] = (int32_t) (z - two24 * fw);
		z = q[j - 1] + fw;
	}

	/* compute n */
	z = scalbn(z, (int)q0);				/* actual value of z */
	z -= 8.0 * floor(z * 0.125);		/* trim off integer >= 8 */
	n = (int32_t) z;
	z -= (double) n;
	ih = 0;
	if (q0 > 0)
	{									/* need iq[jz-1] to determine n */
		i = (iq[jz - 1] >> (24 - q0));
		n += i;
		iq[jz - 1] -= i << (24 - q0);
		ih = iq[jz - 1] >> (23 - q0);
	} else if (q0 == 0)
	{
		ih = iq[jz - 1] >> 23;
	} else if (z >= 0.5)
	{
		ih = 2;
	}

	if (ih > 0)
	{									/* q > 0.5 */
		n += 1;
		carry = 0;
		for (i = 0; i < jz; i++)
		{								/* compute 1-q */
			j = iq[i];
			if (carry == 0)
			{
				if (j != 0)
				{
					carry = 1;
					iq[i] = IC(0x1000000) - j;
				}
			} else
				iq[i] = IC(0xffffff) - j;
		}
		if (q0 > 0)
		{								/* rare case: chance is 1 in 12 */
			switch ((int)q0)
			{
			case 1:
				iq[jz - 1] &= IC(0x7fffff);
				break;
			case 2:
				iq[jz - 1] &= IC(0x3fffff);
				break;
			}
		}
		if (ih == 2)
		{
			z = one - z;
			if (carry != 0)
				z -= scalbn(one, (int)q0);
		}
	}

	/* check if recomputation is needed */
	if (z == zero)
	{
		j = 0;
		for (i = jz - 1; i >= jk; i--)
			j |= iq[i];
		if (j == 0)
		{								/* need recomputation */
			for (k = 1; iq[jk - k] == 0; k++)	/* k = no. of terms needed */
				;
			for (i = jz + 1; i <= jz + k; i++)
			{							/* add q[jz+1] to q[jz+k] */
				f[jx + i] = (double) two_over_pi[jv + i];
				for (j = 0, fw = 0.0; j <= jx; j++)
					fw += x[j] * f[jx + i - j];
				q[i] = fw;
			}
			jz += k;
			goto recompute;
		}
	}

	/* chop off zero terms */
	if (z == 0.0)
	{
		jz -= 1;
		q0 -= 24;
		while (iq[jz] == 0)
		{
			jz--;
			q0 -= 24;
		}
	} else
	{									/* break z into 24-bit if necessary */
		z = scalbn(z, (int)-q0);
		if (z >= two24)
		{
			fw = (double) ((int32_t) (twon24 * z));
			iq[jz] = (int32_t) (z - two24 * fw);
			jz += 1;
			q0 += 24;
			iq[jz] = (int32_t) fw;
		} else
		{
			iq[jz] = (int32_t) z;
		}
	}

	/* convert integer "bit" chunk to floating-point value */
	fw = scalbn(one, (int)q0);
	for (i = jz; i >= 0; i--)
	{
		q[i] = fw * (double) iq[i];
		fw *= twon24;
	}

	/* compute PIo2[0,...,jp]*q[jz,...,0] */
	for (i = jz; i >= 0; i--)
	{
		for (fw = 0.0, k = 0; k <= jp && k <= jz - i; k++)
			fw += PIo2[k] * q[i + k];
		fq[jz - i] = fw;
	}

	/* compress fq[] into y[] */
	switch (prec)
	{
	case 0:
		fw = 0.0;
		for (i = jz; i >= 0; i--)
			fw += fq[i];
		y[0] = (ih == 0) ? fw : -fw;
		break;
	case 1:
	case 2:
		{
			volatile double fv = 0.0;

			for (i = jz; i >= 0; i--)
				fv += fq[i];
			y[0] = (ih == 0) ? fv : -fv;
			fv = fq[0] - fv;
			for (i = 1; i <= jz; i++)
				fv += fq[i];
			y[1] = (ih == 0) ? fv : -fv;
		}
		break;
	case 3:							/* painful */
		for (i = jz; i > 0; i--)
		{
			volatile double fv = (double) (fq[i - 1] + fq[i]);

			fq[i] += fq[i - 1] - fv;
			fq[i - 1] = fv;
		}
		for (i = jz; i > 1; i--)
		{
			volatile double fv = (double) (fq[i - 1] + fq[i]);

			fq[i] += fq[i - 1] - fv;
			fq[i - 1] = fv;
		}
		for (fw = 0.0, i = jz; i >= 2; i--)
			fw += fq[i];
		if (ih == 0)
		{
			y[0] = fq[0];
			y[1] = fq[1];
			y[2] = fw;
		} else
		{
			y[0] = -fq[0];
			y[1] = -fq[1];
			y[2] = -fw;
		}
	}
	return n & 7;
}

int32_t __ieee754_rem_pio2(double x, double *y)
{
	double z, w, t, r, fn;
	double tx[3];
	int32_t e0, i, j, nx, n, ix, hx;
	uint32_t low;

	static const int32_t npio2_hw[] = {
		IC(0x3FF921FB), IC(0x400921FB), IC(0x4012D97C), IC(0x401921FB), IC(0x401F6A7A), IC(0x4022D97C),
		IC(0x4025FDBB), IC(0x402921FB), IC(0x402C463A), IC(0x402F6A7A), IC(0x4031475C), IC(0x4032D97C),
		IC(0x40346B9C), IC(0x4035FDBB), IC(0x40378FDB), IC(0x403921FB), IC(0x403AB41B), IC(0x403C463A),
		IC(0x403DD85A), IC(0x403F6A7A), IC(0x40407E4C), IC(0x4041475C), IC(0x4042106C), IC(0x4042D97C),
		IC(0x4043A28C), IC(0x40446B9C), IC(0x404534AC), IC(0x4045FDBB), IC(0x4046C6CB), IC(0x40478FDB),
		IC(0x404858EB), IC(0x404921FB)
	};

	/*
	 * invpio2:  53 bits of 2/pi
	 * pio2_1:   first  33 bit of pi/2
	 * pio2_1t:  pi/2 - pio2_1
	 * pio2_2:   second 33 bit of pi/2
	 * pio2_2t:  pi/2 - (pio2_1+pio2_2)
	 * pio2_3:   third  33 bit of pi/2
	 * pio2_3t:  pi/2 - (pio2_1+pio2_2+pio2_3)
	 */

	static const double zero = 0.00000000000000000000e+00;	/* 0x00000000, 0x00000000 */
	static const double half = 5.00000000000000000000e-01;	/* 0x3FE00000, 0x00000000 */
	static const double two24 = 1.67772160000000000000e+07;	/* 0x41700000, 0x00000000 */
	static const double invpio2 = 6.36619772367581382433e-01;	/* 0x3FE45F30, 0x6DC9C883 */
	static const double pio2_1 = 1.57079632673412561417e+00;	/* 0x3FF921FB, 0x54400000 */
	static const double pio2_1t = 6.07710050650619224932e-11;	/* 0x3DD0B461, 0x1A626331 */
	static const double pio2_2 = 6.07710050630396597660e-11;	/* 0x3DD0B461, 0x1A600000 */
	static const double pio2_2t = 2.02226624879595063154e-21;	/* 0x3BA3198A, 0x2E037073 */
	static const double pio2_3 = 2.02226624871116645580e-21;	/* 0x3BA3198A, 0x2E000000 */
	static const double pio2_3t = 8.47842766036889956997e-32;	/* 0x397B839A, 0x252049C1 */

	GET_HIGH_WORD(hx, x);				/* high word of x */
	ix = hx & IC(0x7fffffff);
	if (ix <= IC(0x3fe921fb))			/* |x| ~<= pi/4 , no need for reduction */
	{
		y[0] = x;
		y[1] = 0;
		return 0;
	}

	if (ix < IC(0x4002d97c))
	{									/* |x| < 3pi/4, special case with n=+-1 */
		if (hx > 0)
		{
			z = x - pio2_1;
			if (ix != IC(0x3ff921fb))
			{							/* 33+53 bit pi is good enough */
				y[0] = z - pio2_1t;
				y[1] = (z - y[0]) - pio2_1t;
			} else
			{							/* near pi/2, use 33+33+53 bit pi */
				z -= pio2_2;
				y[0] = z - pio2_2t;
				y[1] = (z - y[0]) - pio2_2t;
			}
			return 1;
		} else
		{								/* negative x */
			z = x + pio2_1;
			if (ix != IC(0x3ff921fb))
			{							/* 33+53 bit pi is good enough */
				y[0] = z + pio2_1t;
				y[1] = (z - y[0]) + pio2_1t;
			} else
			{							/* near pi/2, use 33+33+53 bit pi */
				z += pio2_2;
				y[0] = z + pio2_2t;
				y[1] = (z - y[0]) + pio2_2t;
			}
			return -1;
		}
	}

	if (ix <= IC(0x413921fb))
	{									/* |x| ~<= 2^19*(pi/2), medium size */
		t = fabs(x);
		n = (int32_t) (t * invpio2 + half);
		fn = (double) n;
		r = t - fn * pio2_1;
		w = fn * pio2_1t;				/* 1st round good to 85 bit */
		if (n < 32 && ix != npio2_hw[n - 1])
		{
			y[0] = r - w;				/* quick check no cancellation */
		} else
		{
			uint32_t high;

			j = ix >> IEEE754_DOUBLE_SHIFT;
			y[0] = r - w;
			GET_HIGH_WORD(high, y[0]);
			i = j - ((high >> IEEE754_DOUBLE_SHIFT) & IEEE754_DOUBLE_MAXEXP);
			if (i > 16)
			{							/* 2nd iteration needed, good to 118 */
				t = r;
				w = fn * pio2_2;
				r = t - w;
				w = fn * pio2_2t - ((t - r) - w);
				y[0] = r - w;
				GET_HIGH_WORD(high, y[0]);
				i = j - ((high >> IEEE754_DOUBLE_SHIFT) & IEEE754_DOUBLE_MAXEXP);
				if (i > 49)
				{						/* 3rd iteration need, 151 bits acc */
					t = r;				/* will cover all possible cases */
					w = fn * pio2_3;
					r = t - w;
					w = fn * pio2_3t - ((t - r) - w);
					y[0] = r - w;
				}
			}
		}
		y[1] = (r - y[0]) - w;
		if (hx < 0)
		{
			y[0] = -y[0];
			y[1] = -y[1];
			return -n;
		} else
			return n;
	}

	/*
	 * all other (large) arguments
	 */
	if (ix >= IC(0x7ff00000))
	{									/* x is inf or NaN */
		y[0] = y[1] = x - x;
		return 0;
	}
	/* set z = scalbn(|x|,ilogb(x)-23) */
	GET_LOW_WORD(low, x);
	e0 = (ix >> IEEE754_DOUBLE_SHIFT) - (IEEE754_DOUBLE_BIAS + 23);			/* e0 = ilogb(z)-23; */
	INSERT_WORDS(z, ix - (e0 << IEEE754_DOUBLE_SHIFT), low);
	for (i = 0; i < 2; i++)
	{
		tx[i] = (double) ((int32_t) (z));
		z = (z - tx[i]) * two24;
	}
	tx[2] = z;
	nx = 3;
	while (tx[nx - 1] == zero)
		nx--;							/* skip zero term */
	n = __kernel_rem_pio2(tx, y, e0, nx, 2);
	if (hx < 0)
	{
		y[0] = -y[0];
		y[1] = -y[1];
		return -n;
	}
	return n;
}

extern "C" {

  int c_rem_pio2(const double a, double* b) {
    int n = __ieee754_rem_pio2(a, b);
    return (int)(n & 3);
  }

/* add */
void c_dd_add(const double *a, const double *b, double *c) {
  dd_real cc;
  cc = dd_real(a) + dd_real(b);
  TO_DOUBLE_PTR(cc, c);
}
void c_dd_add_dd_d(const double *a, double b, double *c) {
  dd_real cc;
  cc = dd_real(a) + b;
  TO_DOUBLE_PTR(cc, c);
}
void c_dd_add_d_dd(double a, const double *b, double *c) {
  dd_real cc;
  cc = a + dd_real(b);
  TO_DOUBLE_PTR(cc, c);
}


/* sub */
void c_dd_sub(const double *a, const double *b, double *c) {
  dd_real cc;
  cc = dd_real(a) - dd_real(b);
  TO_DOUBLE_PTR(cc, c);
}
void c_dd_sub_dd_d(const double *a, double b, double *c) {
  dd_real cc;
  cc = dd_real(a) - b;
  TO_DOUBLE_PTR(cc, c);
}
void c_dd_sub_d_dd(double a, const double *b, double *c) {
  dd_real cc;
  cc = a - dd_real(b);
  TO_DOUBLE_PTR(cc, c);
}


/* mul */
void c_dd_mul(const double *a, const double *b, double *c) {
  dd_real cc;
  cc = dd_real(a) * dd_real(b);
  TO_DOUBLE_PTR(cc, c);
}
void c_dd_mul_dd_d(const double *a, double b, double *c) {
  dd_real cc;
  cc = dd_real(a) * b;
  TO_DOUBLE_PTR(cc, c);
}
void c_dd_mul_d_dd(double a, const double *b, double *c) {
  dd_real cc;
  cc = a * dd_real(b);
  TO_DOUBLE_PTR(cc, c);
}


/* div */
void c_dd_div(const double *a, const double *b, double *c) {
  dd_real cc;
  cc = dd_real(a) / dd_real(b);
  TO_DOUBLE_PTR(cc, c);
}
void c_dd_div_dd_d(const double *a, double b, double *c) {
  dd_real cc;
  cc = dd_real(a) / b;
  TO_DOUBLE_PTR(cc, c);
}
void c_dd_div_d_dd(double a, const double *b, double *c) {
  dd_real cc;
  cc = a / dd_real(b);
  TO_DOUBLE_PTR(cc, c);
}


/* copy */
void c_dd_copy(const double *a, double *b) {
  b[0] = a[0];
  b[1] = a[1];
}
void c_dd_copy_d(double a, double *b) {
  b[0] = a;
  b[1] = 0.0;
}


void c_dd_sqrt(const double *a, double *b) {
  dd_real bb;
  bb = sqrt(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}
void c_dd_sqr(const double *a, double *b) {
  dd_real bb;
  bb = sqr(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}

void c_dd_abs(const double *a, double *b) {
  dd_real bb;
  bb = abs(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}

void c_dd_npwr(const double *a, int n, double *b) {
  dd_real bb;
  bb = npwr(dd_real(a), n);
  TO_DOUBLE_PTR(bb, b);
}

void c_dd_nroot(const double *a, int n, double *b) {
  dd_real bb;
  bb = nroot(dd_real(a), n);
  TO_DOUBLE_PTR(bb, b);
}

void c_dd_nint(const double *a, double *b) {
  dd_real bb;
  bb = nint(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}
void c_dd_aint(const double *a, double *b) {
  dd_real bb;
  bb = aint(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}
void c_dd_floor(const double *a, double *b) {
  dd_real bb;
  bb = floor(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}
void c_dd_ceil(const double *a, double *b) {
  dd_real bb;
  bb = ceil(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}

void c_dd_log(const double *a, double *b) {
  dd_real bb;
  bb = log(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}
void c_dd_log10(const double *a, double *b) {
  dd_real bb;
  bb = log10(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}
void c_dd_exp(const double *a, double *b) {
  dd_real bb;
  bb = exp(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}

void c_dd_sin(const double *a, double *b) {
  dd_real bb;
  bb = sin(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}
void c_dd_cos(const double *a, double *b) {
  dd_real bb;
  bb = cos(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}
void c_dd_tan(const double *a, double *b) {
  dd_real bb;
  bb = tan(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}

void c_dd_asin(const double *a, double *b) {
  dd_real bb;
  bb = asin(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}
void c_dd_acos(const double *a, double *b) {
  dd_real bb;
  bb = acos(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}
void c_dd_atan(const double *a, double *b) {
  dd_real bb;
  bb = atan(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}

void c_dd_atan2(const double *a, const double *b, double *c) {
  dd_real cc;
  cc = atan2(dd_real(a), dd_real(b));
  TO_DOUBLE_PTR(cc, c);
}

void c_dd_sinh(const double *a, double *b) {
  dd_real bb;
  bb = sinh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}
void c_dd_cosh(const double *a, double *b) {
  dd_real bb;
  bb = cosh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}
void c_dd_tanh(const double *a, double *b) {
  dd_real bb;
  bb = tanh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}

void c_dd_asinh(const double *a, double *b) {
  dd_real bb;
  bb = asinh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}
void c_dd_acosh(const double *a, double *b) {
  dd_real bb;
  bb = acosh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}
void c_dd_atanh(const double *a, double *b) {
  dd_real bb;
  bb = atanh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
}

void c_dd_sincos(const double *a, double *s, double *c) {
  dd_real ss, cc;
  sincos(dd_real(a), ss, cc);
  TO_DOUBLE_PTR(ss, s);
  TO_DOUBLE_PTR(cc, c);
}

void c_dd_sincosh(const double *a, double *s, double *c) {
  dd_real ss, cc;
  sincosh(dd_real(a), ss, cc);
  TO_DOUBLE_PTR(ss, s);
  TO_DOUBLE_PTR(cc, c);
}

void c_dd_read(const char *s, double *a) {
  dd_real aa(s);
  TO_DOUBLE_PTR(aa, a);
}

void c_dd_swrite(const double *a, int precision, char *s, int len) {
  dd_real(a).write(s, len, precision);
}

void c_dd_write(const double *a) {
  std::cout << dd_real(a).to_string(dd_real::_ndigits) << std::endl;
}

void c_dd_neg(const double *a, double *b) {
  b[0] = -a[0];
  b[1] = -a[1];
}

void c_dd_rand(double *a) {
  dd_real aa;
  aa = ddrand();
  TO_DOUBLE_PTR(aa, a);
}

void c_dd_comp(const double *a, const double *b, int *result) {
  dd_real aa(a), bb(b);
  if (aa < bb)
    *result = -1;
  else if (aa > bb)
    *result = 1;
  else 
    *result = 0;
}

void c_dd_comp_dd_d(const double *a, double b, int *result) {
  dd_real aa(a), bb(b);
  if (aa < bb)
    *result = -1;
  else if (aa > bb)
    *result = 1;
  else 
    *result = 0;
}

void c_dd_comp_d_dd(double a, const double *b, int *result) {
  dd_real aa(a), bb(b);
  if (aa < bb)
    *result = -1;
  else if (aa > bb)
    *result = 1;
  else 
    *result = 0;
}

void c_dd_pi(double *a) {
  TO_DOUBLE_PTR(dd_real::_pi, a);
}

}

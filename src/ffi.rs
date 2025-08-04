// src/ffi.rs
#![allow(non_camel_case_types)]

use libc::{c_double, c_float, c_int, size_t};

pub type vDSP_Stride  = isize;
pub type vDSP_Length  = size_t;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    /* ---------- vDSP (vector operations) ---------- */
    // single-precision
    pub fn vDSP_dotpr(
        a: *const c_float,
        ia: vDSP_Stride,
        b: *const c_float,
        ib: vDSP_Stride,
        c: *mut c_float,
        n: vDSP_Length,
    );
    pub fn vDSP_svesq(
        a: *const c_float,
        ia: vDSP_Stride,
        c: *mut c_float,
        n: vDSP_Length,
    );
    // double-precision
    pub fn vDSP_dotprD(
        a: *const c_double,
        ia: vDSP_Stride,
        b: *const c_double,
        ib: vDSP_Stride,
        c: *mut c_double,
        n: vDSP_Length,
    );
    pub fn vDSP_svesqD(
        a: *const c_double,
        ia: vDSP_Stride,
        c: *mut c_double,
        n: vDSP_Length,
    );

    /* ---------- CBLAS (matrix multiply) ---------- */
    pub fn cblas_sgemm(
        order: c_int,
        trans_a: c_int,
        trans_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        b: *const c_float,
        ldb: c_int,
        beta: c_float,
        c: *mut c_float,
        ldc: c_int,
    );
    pub fn cblas_dgemm(
        order: c_int,
        trans_a: c_int,
        trans_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_double,
        a: *const c_double,
        lda: c_int,
        b: *const c_double,
        ldb: c_int,
        beta: c_double,
        c: *mut c_double,
        ldc: c_int,
    );
}
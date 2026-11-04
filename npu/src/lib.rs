// SPDX-License-Identifier: MIT
//! C ABI over hipfire's XDNA2 G80 BF16 trunk transport (`railgun::npu::trunk`), for the
//! Qwen3.8-Flash-Next prefill input projections (GDN in_proj_qkvz + in_proj_ba, QSA
//! q/k/v + indexer) that libstrix moves to the NPU.
//!
//! Flow (python/strix_vllm/npu.py drives it):
//!
//! 1. `strix_npu_builder_new(rows_max)`, then `strix_npu_builder_add_layer` once per layer
//!    with its fused bf16 weight `[n_live, 2560]` (vLLM's `[out, in]` layout). The weight
//!    is quantized here exactly as hipfire packs its NPU panels: per output row
//!    `ws = grid15_up(absmax / 127)`, `w8 = clamp(round_ties_even(w / ws), -127, 127)`.
//! 2. `strix_npu_open` builds one row-major G80 program per shape, opens the NPU
//!    hardware context, pins the panels and the arenas, and starts the relay thread.
//! 3. Per kick, on the producer's HIP stream: `prepare_bank`, pack A into the A arena
//!    (GPU kernel), `kick`, `wait_gpu`, read C, `retire`.
//!
//! Every function returns 0 on success and -1 on failure; `strix_npu_last_error` has
//! the message of the last failure on this thread.

use core::ffi::{c_char, c_int, c_void};
use peacemaker_npu::kernels::{gemm_g80, w8a8_epi::grid15_up};
use railgun::npu::trunk::{
    LayerPanels, NpuTrunkRuntime, OpenOptions, PanelBuild, PanelProgram, Seq, Shape,
    TxnBuilder, PANEL_K, PANEL_N,
};
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::sync::Arc;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn fail(msg: impl Into<String>) -> c_int {
    let msg = msg.into().replace('\0', " ");
    LAST_ERROR.with(|e| *e.borrow_mut() = CString::new(msg).unwrap_or_default());
    -1
}

/// Message of this thread's last failure (empty if none). Valid until the next failure.
#[no_mangle]
pub extern "C" fn strix_npu_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

fn shape_of(id: c_int) -> Result<Shape, String> {
    match id {
        0 => Ok(Shape::GdnIn),
        1 => Ok(Shape::QsaIn),
        _ => Err(format!("shape id {id}: expected 0 (GdnIn) or 1 (QsaIn)")),
    }
}

fn design(rows: usize, shape: Shape) -> Result<peacemaker_npu::kernels::gemm_array::ArrayDesign, String> {
    gemm_g80::build_panel_program_rowmajor(rows, shape.pitch(), 0)
}

pub struct Builder {
    rows_max: usize,
    layers: Vec<LayerPanels>,
}

/// Starts a weight set for programs of up to `rows_max` rows (1..=65536).
#[no_mangle]
pub extern "C" fn strix_npu_builder_new(rows_max: usize) -> *mut Builder {
    if !(1..=65536).contains(&rows_max) {
        fail(format!("rows_max {rows_max} outside 1..=65536"));
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(Builder { rows_max, layers: Vec::new() }))
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// One output row: hipfire's direct-BF16 W8 quantizer (npu_trunk/packing.rs).
fn quantize_row(row: &[u16], out: &mut [i8]) -> (u16, i16) {
    let w: Vec<f32> = row.iter().map(|&b| bf16_to_f32(b)).collect();
    let absmax = w.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    let raw = ((absmax as f64) / 127.0) as f32;
    let grid = grid15_up(raw);
    let ws = (f64::from(grid.0) * 2f64.powi(i32::from(grid.1))) as f32;
    if ws == 0.0 {
        out.fill(0);
    } else {
        for (o, &x) in out.iter_mut().zip(w.iter()) {
            *o = ((x / ws).round_ties_even() as i32).clamp(-127, 127) as i8;
        }
    }
    grid
}

fn add_layer(b: &mut Builder, layer: usize, shape: Shape, w: &[u16], n_live: usize) -> Result<(), String> {
    let k = PANEL_K;
    if n_live == 0 || n_live > shape.pitch() {
        return Err(format!("{shape:?}: {n_live} live columns, at most {}", shape.pitch()));
    }
    if b.layers.iter().any(|l| l.layer == layer && l.shape == shape) {
        return Err(format!("layer {layer} {shape:?} added twice"));
    }
    // Quantize every live output row, in parallel (row-independent).
    let mut codes = vec![0i8; n_live * k];
    let mut scales = vec![(0u16, 0i16); n_live];
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get()).min(32);
    let per = n_live.div_ceil(threads);
    std::thread::scope(|s| {
        for ((cchunk, schunk), wchunk) in codes
            .chunks_mut(per * k)
            .zip(scales.chunks_mut(per))
            .zip(w.chunks(per * k))
        {
            s.spawn(move || {
                for ((c, sc), row) in cchunk.chunks_mut(k).zip(schunk.iter_mut()).zip(wchunk.chunks(k)) {
                    *sc = quantize_row(row, c);
                }
            });
        }
    });
    let d = design(b.rows_max, shape)?;
    let panels = shape.pitch() / PANEL_N;
    let mut out = Vec::with_capacity(panels);
    for p in 0..panels {
        // pack_b_bf16 takes the panel K-major: b[kk * 1280 + n], zero past the live columns.
        let mut bp = vec![0i8; k * PANEL_N];
        let mut sp = vec![(0u16, 0i16); PANEL_N];
        for n in 0..PANEL_N {
            let col = p * PANEL_N + n;
            if col >= n_live {
                break;
            }
            sp[n] = scales[col];
            let src = &codes[col * k..(col + 1) * k];
            for (kk, &v) in src.iter().enumerate() {
                bp[kk * PANEL_N + n] = v;
            }
        }
        out.push(gemm_g80::pack_b_bf16(&d, &bp, &sp));
    }
    b.layers.push(LayerPanels { layer, shape, panels: out });
    Ok(())
}

/// Adds layer `layer`'s fused weight for `shape` (0 GdnIn, 1 QsaIn): `n_live` rows of
/// 2560 bf16 values (raw bits), row-major. Columns past `n_live` up to the shape's pitch
/// (16640 / 14080) are zero.
///
/// # Safety
/// `b` comes from `strix_npu_builder_new`; `weight` points at `n_live * 2560` u16 values.
#[no_mangle]
pub unsafe extern "C" fn strix_npu_builder_add_layer(
    b: *mut Builder,
    layer: usize,
    shape: c_int,
    weight: *const u16,
    n_live: usize,
) -> c_int {
    let Some(b) = (unsafe { b.as_mut() }) else { return fail("null builder") };
    if weight.is_null() {
        return fail("null weight");
    }
    let shape = match shape_of(shape) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };
    let w = unsafe { std::slice::from_raw_parts(weight, n_live * PANEL_K) };
    match add_layer(b, layer, shape, w, n_live) {
        Ok(()) => 0,
        Err(e) => fail(e),
    }
}

/// Frees a builder that was not opened.
///
/// # Safety
/// `b` comes from `strix_npu_builder_new` and is not used again.
#[no_mangle]
pub unsafe extern "C" fn strix_npu_builder_free(b: *mut Builder) {
    if !b.is_null() {
        drop(unsafe { Box::from_raw(b) });
    }
}

pub struct Runtime {
    rt: NpuTrunkRuntime,
}

/// Opens the NPU transport with the builder's layers (consumes the builder, also on
/// failure). `pci` is the producer GPU's PCI bus id (e.g. "0000:c6:00.0"); `banks` is
/// the number of independent A/C arena sets; `watchdog_ms` bounds one chain's NPU time
/// (0: railgun's 300 ms, at most 3000).
///
/// # Safety
/// `b` comes from `strix_npu_builder_new`; `pci` is a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn strix_npu_open(
    b: *mut Builder,
    pci: *const c_char,
    banks: usize,
    watchdog_ms: u64,
) -> *mut Runtime {
    if b.is_null() || pci.is_null() {
        fail("null builder or pci");
        return std::ptr::null_mut();
    }
    let b = unsafe { Box::from_raw(b) };
    let pci = match unsafe { CStr::from_ptr(pci) }.to_str() {
        Ok(s) => s.to_owned(),
        Err(_) => {
            fail("pci is not UTF-8");
            return std::ptr::null_mut();
        }
    };
    let result = (|| -> Result<Runtime, String> {
        let mut programs = Vec::new();
        for shape in [Shape::GdnIn, Shape::QsaIn] {
            if !b.layers.iter().any(|l| l.shape == shape) {
                continue;
            }
            let d = design(b.rows_max, shape)?;
            programs.push(PanelProgram {
                shape,
                pdi: d.pdi.clone(),
                txn: d.insts.clone(),
                padded_rows: b.rows_max.div_ceil(256) * 256,
                pitch: shape.pitch(),
            });
        }
        if programs.is_empty() {
            return Err("no layers added".into());
        }
        // Row-adaptive chains: a kick of fewer rows runs a program of its own wave count,
        // built on the relay thread the first time and cached.
        let builder: TxnBuilder = Arc::new(|shape: Shape, rows: usize| {
            let d = design(rows, shape)?;
            Ok(PanelBuild { pdi: d.pdi, txn: d.insts })
        });
        let rt = NpuTrunkRuntime::open_with_builder(
            programs,
            b.layers,
            banks,
            b.rows_max,
            &pci,
            OpenOptions {
                watchdog: if watchdog_ms == 0 {
                    OpenOptions::default().watchdog
                } else {
                    std::time::Duration::from_millis(watchdog_ms)
                },
                ..OpenOptions::default()
            },
            Some(builder),
        )?;
        Ok(Runtime { rt })
    })();
    match result {
        Ok(rt) => Box::into_raw(Box::new(rt)),
        Err(e) => {
            fail(e);
            std::ptr::null_mut()
        }
    }
}

macro_rules! rt {
    ($p:expr) => {
        match unsafe { ($p as *const Runtime).as_ref() } {
            Some(r) => &r.rt,
            None => return fail("null runtime"),
        }
    };
}

/// An arena: `kind` 0 = the packed-A arena of `bank` (`shape` ignored), 1 = the BF16 C
/// arena of (`bank`, `shape`). Writes its GPU address, host address and length.
///
/// # Safety
/// `rt` comes from `strix_npu_open`; the out pointers are valid.
#[no_mangle]
pub unsafe extern "C" fn strix_npu_arena(
    rt: *const Runtime,
    kind: c_int,
    bank: usize,
    shape: c_int,
    device: *mut u64,
    host: *mut *mut c_void,
    len: *mut usize,
) -> c_int {
    let r = rt!(rt);
    let reg = match kind {
        0 => r.a_arena(bank),
        1 => match shape_of(shape) {
            Ok(s) => r.c_arena(bank, s),
            Err(e) => return fail(e),
        },
        _ => return fail(format!("arena kind {kind}")),
    };
    match reg {
        Ok(h) => {
            unsafe {
                *device = h.device;
                *host = h.host.cast();
                *len = h.len;
            }
            0
        }
        Err(e) => fail(e),
    }
}

/// Padded rows a `rows`-row kick of `shape` packs A for and runs (0 on error).
///
/// # Safety
/// `rt` comes from `strix_npu_open`.
#[no_mangle]
pub unsafe extern "C" fn strix_npu_chain_rows(rt: *const Runtime, shape: c_int, rows: usize) -> usize {
    let Some(r) = (unsafe { rt.as_ref() }) else { return 0 };
    shape_of(shape).map_or(0, |s| r.rt.chain_rows(s, rows))
}

/// Pitch (elements) of a shape's C arena rows.
#[no_mangle]
pub extern "C" fn strix_npu_pitch(shape: c_int) -> usize {
    shape_of(shape).map_or(0, |s| s.pitch())
}

/// Enqueues the bank's retirement on `stream` before the producer writes A.
///
/// # Safety
/// `rt` comes from `strix_npu_open`; `stream` is a live HIP stream of the producer GPU.
#[no_mangle]
pub unsafe extern "C" fn strix_npu_prepare_bank(rt: *const Runtime, bank: usize, stream: *mut c_void) -> c_int {
    let r = rt!(rt);
    match unsafe { r.prepare_bank(bank, stream) } {
        Ok(()) => 0,
        Err(e) => fail(e),
    }
}

/// Records a release after the A writes on `stream` and queues layer `layer`'s chain of
/// `shape` over `rows` rows of `bank`. Writes the sequence value to wait on.
///
/// # Safety
/// As `strix_npu_prepare_bank`; `seq` is valid.
#[no_mangle]
pub unsafe extern "C" fn strix_npu_kick(
    rt: *const Runtime,
    layer: usize,
    shape: c_int,
    bank: usize,
    rows: usize,
    stream: *mut c_void,
    seq: *mut u32,
) -> c_int {
    let r = rt!(rt);
    let shape = match shape_of(shape) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };
    match unsafe { r.kick(layer, shape, bank, rows, stream) } {
        Ok(s) => {
            unsafe { *seq = s.value };
            0
        }
        Err(e) => fail(e),
    }
}

/// Enqueues on `stream` the wait for (`bank`, `seq`) before the C consumers. A failed chain
/// also releases the wait: call `strix_npu_check_error` before trusting C.
///
/// # Safety
/// As `strix_npu_prepare_bank`.
#[no_mangle]
pub unsafe extern "C" fn strix_npu_wait_gpu(rt: *const Runtime, bank: usize, seq: u32, stream: *mut c_void) -> c_int {
    let r = rt!(rt);
    match unsafe { r.wait_gpu(Seq { bank, value: seq }, stream) } {
        Ok(()) => 0,
        Err(e) => fail(e),
    }
}

/// After the last C consumer of (`bank`, `seq`) is enqueued on `stream`.
///
/// # Safety
/// As `strix_npu_prepare_bank`.
#[no_mangle]
pub unsafe extern "C" fn strix_npu_retire(rt: *const Runtime, bank: usize, seq: u32, stream: *mut c_void) -> c_int {
    let r = rt!(rt);
    match unsafe { r.retire_bank(Seq { bank, value: seq }, stream) } {
        Ok(()) => 0,
        Err(e) => fail(e),
    }
}

/// Blocks the host until (`bank`, `seq`) completes; writes the NPU busy time (ns).
///
/// # Safety
/// `rt` comes from `strix_npu_open`; `busy_ns` is valid or null.
#[no_mangle]
pub unsafe extern "C" fn strix_npu_wait_host(rt: *const Runtime, bank: usize, seq: u32, busy_ns: *mut u64) -> c_int {
    let r = rt!(rt);
    match r.wait_host(Seq { bank, value: seq }) {
        Ok(c) => {
            if !busy_ns.is_null() {
                unsafe { *busy_ns = c.npu_busy.as_nanos() as u64 };
            }
            0
        }
        Err(e) => fail(e),
    }
}

/// Sticky transport error (nonblocking): -1 with the message if any chain failed.
///
/// # Safety
/// `rt` comes from `strix_npu_open`.
#[no_mangle]
pub unsafe extern "C" fn strix_npu_check_error(rt: *const Runtime) -> c_int {
    let r = rt!(rt);
    match r.check_error() {
        Ok(()) => 0,
        Err(e) => fail(e),
    }
}

/// Checked teardown.
///
/// # Safety
/// `rt` comes from `strix_npu_open` and is not used again.
#[no_mangle]
pub unsafe extern "C" fn strix_npu_close(rt: *mut Runtime) -> c_int {
    if rt.is_null() {
        return 0;
    }
    let r = unsafe { Box::from_raw(rt) };
    match r.rt.close() {
        Ok(()) => 0,
        Err(e) => fail(e),
    }
}

/// Relay statistics of `shape`'s chains completed since the previous call (all shapes are
/// reset by a call): chain count, summed NPU busy time and summed kick-to-done time (ns).
/// Nonblocking.
///
/// # Safety
/// `rt` comes from `strix_npu_open`; `out` points at six u64 (GdnIn, then QsaIn).
#[no_mangle]
pub unsafe extern "C" fn strix_npu_take_stats(rt: *const Runtime, out: *mut u64) -> c_int {
    let r = rt!(rt);
    let s = r.take_chain_stats();
    let out = unsafe { std::slice::from_raw_parts_mut(out, 6) };
    for (i, shape) in [Shape::GdnIn, Shape::QsaIn].into_iter().enumerate() {
        let st = s.shape(shape);
        out[3 * i] = st.chains;
        out[3 * i + 1] = st.npu_busy_ns;
        out[3 * i + 2] = st.kick_to_done_ns;
    }
    0
}

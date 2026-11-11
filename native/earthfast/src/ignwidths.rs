//! IGN BD TOPO carriageway widths matched to a cell's roads (server/ign_largeur.py's join is the glue): per road, its
//! longest ring sampled at SAMPLES points, each the width of the nearest BD TOPO segment running along it, and the
//! median of what was found. Python did it road by road (~0.06 CPU-s a cell; a city's first block planned ~160
//! cells: 11 CPU-s of its 33 s, 2026-11-11). The very same arithmetic, in the same order (numpy's pairwise sum for the
//! ring's mean latitude, a running sum for its lengths, the grid's cells in segment order, ties to the later segment),
//! so the widths are the same.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;

/// numpy's pairwise summation of a float64 run (its add.reduce: numpy/_core/src/umath/loops_utils.h.src).
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

struct Grid {
    cells: HashMap<(i64, i64), Vec<u32>>,
}

impl Grid {
    fn new(s: &numpy::ndarray::Array2<f64>, grid: f64, pad_deg: f64) -> Grid {
        let mut cells: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
        for i in 0..s.nrows() {
            let (ax, ay, bx, by) = (s[[i, 0]], s[[i, 1]], s[[i, 2]], s[[i, 3]]);
            let gx0 = ((ax.min(bx) - pad_deg) * grid).floor() as i64;
            let gx1 = ((ax.max(bx) + pad_deg) * grid).floor() as i64;
            let gy0 = ((ay.min(by) - pad_deg) * grid).floor() as i64;
            let gy1 = ((ay.max(by) + pad_deg) * grid).floor() as i64;
            for gx in gx0..=gx1 {
                for gy in gy0..=gy1 {
                    cells.entry((gx, gy)).or_default().push(i as u32);
                }
            }
        }
        Grid { cells }
    }

    /// The width of the nearest segment running along (ux, uy) at (px, py), or NaN (ign_largeur's former _width_at).
    #[allow(clippy::too_many_arguments)]
    fn width_at(&self, s: &numpy::ndarray::Array2<f64>, px: f64, py: f64, ux: f64, uy: f64, kx: f64, grid: f64,
                match_m: f64, same_way_cos: f64) -> f64 {
        let key = ((px * grid).floor() as i64, (py * grid).floor() as i64);
        let Some(list) = self.cells.get(&key) else { return f64::NAN };
        let mut best = match_m;
        let mut width = f64::NAN;
        for &i in list {
            let i = i as usize;
            let (ax, ay, bx, by, w) = (s[[i, 0]], s[[i, 1]], s[[i, 2]], s[[i, 3]], s[[i, 4]]);
            let dx = (bx - ax) * kx;
            let dy = by - ay;
            let length = dx.hypot(dy);
            if length < 1e-9 || (dx * ux + dy * uy).abs() / length < same_way_cos {
                continue;
            }
            let t = ((((px - ax) * kx) * dx + (py - ay) * dy) / (length * length)).clamp(0.0, 1.0);
            let d = ((px - ax) * kx - t * dx).hypot((py - ay) - t * dy) * 111000.0;
            if d <= best {
                best = d;
                width = w;
            }
        }
        width
    }
}

/// Per candidate road (cand: feature rows of the Features arrays part_off / ring_off / coord_off / lonlat): how many of
/// its `samples` points found a BD TOPO segment (segs (m, 5): ax, ay, bx, by in degrees, width) and their median width
/// (NaN when none). grid: cells per degree; pad_deg: a segment's reach in degrees.
#[pyfunction]
#[pyo3(signature = (segs, part_off, ring_off, coord_off, lonlat, cand, samples, grid, pad_deg, match_m, same_way_cos))]
#[allow(clippy::too_many_arguments)]
pub fn ign_join<'py>(
    py: Python<'py>,
    segs: PyReadonlyArray2<'py, f64>,
    part_off: PyReadonlyArray1<'py, i64>,
    ring_off: PyReadonlyArray1<'py, i64>,
    coord_off: PyReadonlyArray1<'py, i64>,
    lonlat: PyReadonlyArray2<'py, f64>,
    cand: PyReadonlyArray1<'py, i64>,
    samples: usize,
    grid: f64,
    pad_deg: f64,
    match_m: f64,
    same_way_cos: f64,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f64>>)> {
    if segs.as_array().ncols() < 5 || lonlat.as_array().ncols() < 2 || samples == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("ign_join: segs (m, 5), lonlat (k, 2), samples > 0"));
    }
    let s = segs.as_array().to_owned();
    let po = part_off.as_array().to_owned();
    let ro = ring_off.as_array().to_owned();
    let co = coord_off.as_array().to_owned();
    let ll = lonlat.as_array().to_owned();
    let cand = cand.as_array().to_owned();
    let (votes, medians) = py.allow_threads(move || {
        let g = Grid::new(&s, grid, pad_deg);
        let mut votes = vec![0i32; cand.len()];
        let mut medians = vec![f64::NAN; cand.len()];
        let mut lat = Vec::new();
        let mut cum = Vec::new();
        let mut got = Vec::with_capacity(samples);
        for (c, &fi) in cand.iter().enumerate() {
            let fi = fi as usize;
            // its longest ring (the first of the longest), of 2 points or more
            let mut ring: Option<(usize, usize)> = None;
            for p in po[fi] as usize..po[fi + 1] as usize {
                for r in ro[p] as usize..ro[p + 1] as usize {
                    let (a, b) = (co[r] as usize, co[r + 1] as usize);
                    let n = b - a;
                    if n >= 2 && ring.map_or(true, |(ra, rb)| n > rb - ra) {
                        ring = Some((a, b));
                    }
                }
            }
            let Some((a, b)) = ring else { continue };
            let n = b - a;
            lat.clear();
            lat.extend((a..b).map(|k| ll[[k, 1]]));
            let kx = (pairwise_sum(&lat) / n as f64).to_radians().cos();
            cum.clear();
            cum.push(0.0f64);
            let mut acc = 0.0f64;
            for k in a..b - 1 {
                acc += ((ll[[k + 1, 0]] - ll[[k, 0]]) * kx).hypot(ll[[k + 1, 1]] - ll[[k, 1]]);
                cum.push(acc);
            }
            let total = cum[n - 1];
            if !(total > 0.0) {
                continue;
            }
            got.clear();
            for i in 0..samples {
                let f = (i as f64 + 0.5) / samples as f64 * total;
                let right = cum.partition_point(|&v| v <= f);       // numpy searchsorted(side="right")
                let j = (right as i64 - 1).min(n as i64 - 2).max(0) as usize;
                let ds = (cum[j + 1] - cum[j]).max(1e-12);
                let t = (f - cum[j]) / ds;
                let (x0, y0) = (ll[[a + j, 0]], ll[[a + j, 1]]);
                let (dx, dy) = (ll[[a + j + 1, 0]] - x0, ll[[a + j + 1, 1]] - y0);
                let w = g.width_at(&s, x0 + t * dx, y0 + t * dy, dx * kx / ds, dy / ds, kx, grid, match_m,
                                   same_way_cos);
                if !w.is_nan() {
                    got.push(w);
                }
            }
            votes[c] = got.len() as i32;
            if !got.is_empty() {
                got.sort_by(|x, y| x.partial_cmp(y).unwrap());
                let m = got.len();
                medians[c] = if m % 2 == 1 { got[m / 2] } else { (got[m / 2 - 1] + got[m / 2]) / 2.0 };
            }
        }
        (votes, medians)
    });
    Ok((votes.into_pyarray(py), medians.into_pyarray(py)))
}

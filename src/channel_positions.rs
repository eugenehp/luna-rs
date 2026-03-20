//! EEG channel position lookup from embedded standard montage files.
//!
//! Identical to zuna-rs: six MNE-Python ASA `.elc` montage files are embedded
//! at compile time.  Positions are in the ASA head coordinate frame
//! (x=right, y=anterior, z=superior), scaled to head radius 0.085 m.

use std::collections::HashMap;
use std::sync::OnceLock;

const ELC_1020:        &str = include_str!("montages/standard_1020.elc");
const ELC_1005:        &str = include_str!("montages/standard_1005.elc");
const ELC_ALPHABETIC:  &str = include_str!("montages/standard_alphabetic.elc");
const ELC_POSTFIXED:   &str = include_str!("montages/standard_postfixed.elc");
const ELC_PREFIXED:    &str = include_str!("montages/standard_prefixed.elc");
const ELC_PRIMED:      &str = include_str!("montages/standard_primed.elc");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MontageLayout {
    Standard1020,
    Standard1005,
    StandardAlphabetic,
    StandardPostfixed,
    StandardPrefixed,
    StandardPrimed,
}

impl MontageLayout {
    pub const ALL: &'static [MontageLayout] = &[
        MontageLayout::Standard1005,
        MontageLayout::Standard1020,
        MontageLayout::StandardPostfixed,
        MontageLayout::StandardPrimed,
        MontageLayout::StandardPrefixed,
        MontageLayout::StandardAlphabetic,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Standard1020       => "standard_1020",
            Self::Standard1005       => "standard_1005",
            Self::StandardAlphabetic => "standard_alphabetic",
            Self::StandardPostfixed  => "standard_postfixed",
            Self::StandardPrefixed   => "standard_prefixed",
            Self::StandardPrimed     => "standard_primed",
        }
    }
}

pub fn montage_channels(layout: MontageLayout) -> &'static HashMap<String, [f32; 3]> {
    static C1020:  OnceLock<HashMap<String,[f32;3]>> = OnceLock::new();
    static C1005:  OnceLock<HashMap<String,[f32;3]>> = OnceLock::new();
    static CALPHA: OnceLock<HashMap<String,[f32;3]>> = OnceLock::new();
    static CPOST:  OnceLock<HashMap<String,[f32;3]>> = OnceLock::new();
    static CPRE:   OnceLock<HashMap<String,[f32;3]>> = OnceLock::new();
    static CPRIME: OnceLock<HashMap<String,[f32;3]>> = OnceLock::new();

    let (lock, src) = match layout {
        MontageLayout::Standard1020       => (&C1020,  ELC_1020),
        MontageLayout::Standard1005       => (&C1005,  ELC_1005),
        MontageLayout::StandardAlphabetic => (&CALPHA, ELC_ALPHABETIC),
        MontageLayout::StandardPostfixed  => (&CPOST,  ELC_POSTFIXED),
        MontageLayout::StandardPrefixed   => (&CPRE,   ELC_PREFIXED),
        MontageLayout::StandardPrimed     => (&CPRIME, ELC_PRIMED),
    };

    lock.get_or_init(|| parse_elc(src))
}

/// Look up XYZ position (metres) for a channel name. Case-insensitive.
pub fn channel_xyz(name: &str) -> Option<[f32; 3]> {
    let key = normalise(name);
    for &layout in MontageLayout::ALL {
        let map = montage_channels(layout);
        let found = map.iter().find(|(k, _)| normalise(k) == key);
        if let Some((_, &xyz)) = found {
            return Some(xyz);
        }
    }
    None
}

/// For bipolar channels like "FP1-F7", compute the midpoint of the two electrodes.
pub fn bipolar_channel_xyz(name: &str) -> Option<[f32; 3]> {
    if let Some(idx) = name.find('-') {
        let e1 = &name[..idx];
        let e2 = &name[idx + 1..];
        match (channel_xyz(e1), channel_xyz(e2)) {
            (Some(a), Some(b)) => Some([
                (a[0] + b[0]) / 2.0,
                (a[1] + b[1]) / 2.0,
                (a[2] + b[2]) / 2.0,
            ]),
            _ => None,
        }
    } else {
        channel_xyz(name)
    }
}

pub fn nearest_channel(
    target_xyz: [f32; 3],
    candidates: &[([f32; 3], usize)],
) -> Option<usize> {
    candidates.iter()
        .min_by(|(a, _), (b, _)| {
            let da = dist2(*a, target_xyz);
            let db = dist2(*b, target_xyz);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(_, idx)| *idx)
}

/// Normalise a channel name for case-insensitive comparison.
pub fn normalise(name: &str) -> String {
    name.chars()
        .filter(|c| !matches!(c, ' ' | '_'))
        .flat_map(|c| c.to_uppercase())
        .collect()
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0]-b[0]; let dy = a[1]-b[1]; let dz = a[2]-b[2];
    dx*dx + dy*dy + dz*dz
}

fn parse_elc(src: &str) -> HashMap<String, [f32; 3]> {
    const HEAD_SIZE: f32 = 0.085;

    let mm_scale: f32 = {
        let mut s = 1e-3_f32;
        for line in src.lines() {
            if line.contains("UnitPosition") {
                s = if line.contains('m') && !line.contains("mm") { 1.0 } else { 1e-3 };
                break;
            }
        }
        s
    };

    let mut raw: Vec<[f32; 3]> = Vec::new();
    let mut in_pos = false;
    let mut in_lbl = false;
    let mut labels: Vec<String> = Vec::new();

    for line in src.lines() {
        let t = line.trim();
        if t == "Positions" || t.starts_with("Positions") { in_pos = true; in_lbl = false; continue; }
        if t == "Labels"    || t.starts_with("Labels")    { in_lbl = true; in_pos = false; continue; }

        if in_pos {
            let nums: Vec<f32> = if t.contains(':') {
                t.split(':').nth(1).unwrap_or("").split_whitespace()
                    .filter_map(|s| s.parse().ok()).collect()
            } else {
                t.split_whitespace().filter_map(|s| s.parse().ok()).collect()
            };
            if nums.len() == 3 {
                raw.push([nums[0], nums[1], nums[2]]);
            }
        } else if in_lbl && !t.is_empty() {
            labels.push(t.to_string());
        }
    }

    assert_eq!(raw.len(), labels.len());

    let mut pos_m: Vec<[f32; 3]> = raw.iter()
        .map(|p| [p[0] * mm_scale, p[1] * mm_scale, p[2] * mm_scale])
        .collect();

    let mut norms: Vec<f32> = pos_m.iter()
        .map(|p| (p[0]*p[0] + p[1]*p[1] + p[2]*p[2]).sqrt())
        .filter(|&n| n > 1e-6)
        .collect();
    norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = norms[norms.len() / 2];
    if median > 1e-6 {
        let scale = HEAD_SIZE / median;
        for p in &mut pos_m { p[0] *= scale; p[1] *= scale; p[2] *= scale; }
    }

    labels.into_iter().zip(pos_m).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_all_montages() {
        for &layout in MontageLayout::ALL {
            let map = montage_channels(layout);
            assert!(!map.is_empty(), "{} parsed empty", layout.name());
        }
    }

    #[test]
    fn bipolar_midpoint() {
        let xyz = bipolar_channel_xyz("FP1-F7");
        assert!(xyz.is_some(), "FP1-F7 midpoint not found");
        let [x, y, z] = xyz.unwrap();
        assert!(x.abs() <= 0.12 && y.abs() <= 0.12 && z.abs() <= 0.12);
    }

    #[test]
    fn known_channels() {
        for name in &["Cz", "Fz", "Pz", "C3", "C4", "Fp1", "Fp2", "O1", "O2"] {
            assert!(channel_xyz(name).is_some(), "channel '{name}' not found");
        }
    }
}

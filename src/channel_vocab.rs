//! Global channel name vocabulary for LUNA.
//!
//! Python source: `models/modules/channel_embeddings.py`
//!
//! The vocabulary is the sorted union of all channel names across the three
//! datasets used for pre-training and evaluation:
//!   - TUEG (22 bipolar channels)
//!   - Siena (29 unipolar channels)
//!   - SEED-V (62 unipolar channels, from torcheeg SEED_CHANNEL_LIST)
//!
//! Total: 90 unique channel names.
//!
//! The `channel_emb.embeddings.weight` tensor in the checkpoint has shape [90, D].
//! Index `i` corresponds to `CHANNEL_VOCAB[i]`.

/// All 90 channel names in sorted order, matching Python's
/// `sorted(set(TUEG_CHANNEL_LIST + SIENA_CHANNEL_LIST + SEED_CHANNEL_LIST))`.
pub const CHANNEL_VOCAB: &[&str] = &[
    "A1-T3", "AF3", "AF4", "C1", "C2", "C3", "C3-CZ", "C3-P3",
    "C4", "C4-P4", "C4-T4", "C5", "C6", "CB1", "CB2", "CP1",
    "CP2", "CP3", "CP4", "CP5", "CP6", "CPZ", "CZ", "CZ-C4",
    "F1", "F10", "F2", "F3", "F3-C3", "F4", "F4-C4", "F5",
    "F6", "F7", "F7-T3", "F8", "F8-T4", "F9", "FC1", "FC2",
    "FC3", "FC4", "FC5", "FC6", "FCZ", "FP1", "FP1-F3", "FP1-F7",
    "FP2", "FP2-F4", "FP2-F8", "FPZ", "FT7", "FT8", "FZ", "O1",
    "O2", "OZ", "P1", "P2", "P3", "P3-O1", "P4", "P4-O2",
    "P5", "P6", "P7", "P8", "PO3", "PO4", "PO5", "PO6",
    "PO7", "PO8", "POZ", "PZ", "T3", "T3-C3", "T3-T5", "T4",
    "T4-A2", "T4-T6", "T5", "T5-O1", "T6", "T6-O2", "T7", "T8",
    "TP7", "TP8",
];

/// Look up the vocabulary index for a channel name.
/// Returns `None` if the name is not in the vocabulary.
pub fn channel_index(name: &str) -> Option<usize> {
    CHANNEL_VOCAB.iter().position(|&v| v == name)
}

/// Look up vocabulary indices for a list of channel names.
/// Returns `None` for any name not found.
pub fn channel_indices(names: &[&str]) -> Vec<Option<usize>> {
    names.iter().map(|n| channel_index(n)).collect()
}

/// Look up vocabulary indices, panicking on missing names.
pub fn channel_indices_unwrap(names: &[&str]) -> Vec<i64> {
    names.iter().map(|n| {
        channel_index(n)
            .unwrap_or_else(|| panic!("Channel '{}' not in LUNA vocabulary", n)) as i64
    }).collect()
}

/// Get the channel name for a vocabulary index.
pub fn channel_name(index: usize) -> Option<&'static str> {
    CHANNEL_VOCAB.get(index).copied()
}

/// Total vocabulary size (90).
pub const VOCAB_SIZE: usize = 90;

// ── Sub-dataset channel lists ─────────────────────────────────────────────────

/// TUEG 22-channel bipolar montage (TCP).
pub const TUEG_CHANNELS: &[&str] = &[
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "T3-C3", "C3-CZ", "CZ-C4", "C4-T4",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "A1-T3", "T4-A2",
];

/// Siena 29-channel unipolar montage.
pub const SIENA_CHANNELS: &[&str] = &[
    "FP1", "FP2", "F3", "C3", "P3", "O1", "F7", "T3", "T5",
    "FC1", "FC5", "CP1", "CP5", "F9", "FZ", "CZ", "PZ",
    "F4", "C4", "P4", "O2", "F8", "T4", "T6", "FC2", "FC6",
    "CP2", "CP6", "F10",
];

/// SEED-V 62-channel unipolar montage (from torcheeg).
pub const SEED_CHANNELS: &[&str] = &[
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1",
    "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1",
    "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1",
    "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1",
    "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1",
    "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ",
    "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2",
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn vocab_size_is_90() {
        assert_eq!(CHANNEL_VOCAB.len(), VOCAB_SIZE);
    }

    #[test]
    fn vocab_is_sorted() {
        for i in 1..CHANNEL_VOCAB.len() {
            assert!(CHANNEL_VOCAB[i - 1] < CHANNEL_VOCAB[i],
                "CHANNEL_VOCAB not sorted: [{}]='{}' >= [{}]='{}'",
                i - 1, CHANNEL_VOCAB[i - 1], i, CHANNEL_VOCAB[i]);
        }
    }

    #[test]
    fn vocab_matches_python_union() {
        // Verify vocab == sorted(set(TUEG + SIENA + SEED))
        let mut all: BTreeSet<&str> = BTreeSet::new();
        for &ch in TUEG_CHANNELS { all.insert(ch); }
        for &ch in SIENA_CHANNELS { all.insert(ch); }
        for &ch in SEED_CHANNELS { all.insert(ch); }
        let expected: Vec<&str> = all.into_iter().collect();
        assert_eq!(expected.len(), VOCAB_SIZE, "union size mismatch");
        assert_eq!(expected, CHANNEL_VOCAB, "vocab content mismatch");
    }

    #[test]
    fn tueg_indices_valid() {
        for &ch in TUEG_CHANNELS {
            assert!(channel_index(ch).is_some(), "TUEG channel '{}' not in vocab", ch);
        }
    }

    #[test]
    fn siena_indices_valid() {
        for &ch in SIENA_CHANNELS {
            assert!(channel_index(ch).is_some(), "Siena channel '{}' not in vocab", ch);
        }
    }

    #[test]
    fn seed_indices_valid() {
        for &ch in SEED_CHANNELS {
            assert!(channel_index(ch).is_some(), "SEED channel '{}' not in vocab", ch);
        }
    }

    #[test]
    fn round_trip_name_index() {
        for (i, &name) in CHANNEL_VOCAB.iter().enumerate() {
            assert_eq!(channel_index(name), Some(i));
            assert_eq!(channel_name(i), Some(name));
        }
    }
}

//! The intelligence ladder: which blocks a depth rung keeps.
//!
//! Needle 3 is trained so that every depth from 2 to `num_layers` blocks is a
//! deployable subnetwork. Which blocks survive is not a prefix — it is a
//! bisection from both endpoints, so the rungs nest
//! (`S₂ ⊂ S₃ ⊂ … ⊂ S₂₀`) and stay spread across the stack rather than
//! collapsing onto one end.
//!
//! This mirrors upstream's `_ladder_layer_order` exactly. The order is not
//! recorded in the container — `needle build --layers N` writes a file that
//! simply declares the smaller geometry — so a runtime slicing the full model
//! has to reproduce it.
//!
//! For 20 blocks the order is:
//!
//! ```text
//! 0, 19, 9, 14, 4, 6, 11, 16, 2, 7, 12, 17, 1, 3, 5, 8, 10, 13, 15, 18
//! ```

extern crate alloc;
use alloc::vec::Vec;

/// The order blocks are added in as depth grows, longest-gap-first.
///
/// Starts from both endpoints, then repeatedly splits the widest remaining gap
/// at its midpoint. Ties go to the leftmost gap, matching upstream's
/// `key=lambda item: (item[0], -item[1])`.
pub fn ladder_order(num_layers: usize) -> Vec<usize> {
    if num_layers == 0 {
        return Vec::new();
    }
    if num_layers == 1 {
        return alloc::vec![0];
    }
    let mut selected = alloc::vec![0usize, num_layers - 1];
    let mut order = selected.clone();
    while order.len() < num_layers {
        selected.sort_unstable();
        // Widest gap wins; among equals the leftmost, hence `-left` upstream.
        let mut best: Option<(usize, usize, usize)> = None;
        for w in selected.windows(2) {
            let (left, right) = (w[0], w[1]);
            if right - left <= 1 {
                continue;
            }
            let gap = right - left;
            let better = match best {
                None => true,
                Some((bg, bl, _)) => gap > bg || (gap == bg && left < bl),
            };
            if better {
                best = Some((gap, left, right));
            }
        }
        let Some((_, left, right)) = best else { break };
        let mid = (left + right) / 2;
        selected.push(mid);
        order.push(mid);
    }
    order
}

/// The blocks a rung of `depth` keeps, in ascending order.
///
/// Returns `None` when `depth` is outside `2..=num_layers`, which is the range
/// upstream declares deployable.
pub fn ladder_layer_indices(num_layers: usize, depth: usize) -> Option<Vec<usize>> {
    if depth < 2 || depth > num_layers {
        return None;
    }
    let mut kept: Vec<usize> = ladder_order(num_layers).into_iter().take(depth).collect();
    kept.sort_unstable();
    Some(kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_order_matches_upstream_for_twenty_blocks() {
        assert_eq!(
            ladder_order(20),
            alloc::vec![0, 19, 9, 14, 4, 6, 11, 16, 2, 7, 12, 17, 1, 3, 5, 8, 10, 13, 15, 18]
        );
    }

    #[test]
    fn rungs_nest_and_keep_both_endpoints() {
        for depth in 2..=20 {
            let kept = ladder_layer_indices(20, depth).unwrap();
            assert_eq!(kept.len(), depth);
            assert_eq!(kept[0], 0);
            assert_eq!(*kept.last().unwrap(), 19);
            assert!(
                kept.windows(2).all(|w| w[0] < w[1]),
                "ascending, no repeats"
            );
            if depth > 2 {
                let prev = ladder_layer_indices(20, depth - 1).unwrap();
                assert!(
                    prev.iter().all(|l| kept.contains(l)),
                    "depth {depth} must contain every block of {}",
                    depth - 1
                );
            }
        }
    }

    #[test]
    fn a_depth_outside_the_ladder_is_rejected() {
        assert!(ladder_layer_indices(20, 1).is_none());
        assert!(ladder_layer_indices(20, 0).is_none());
        assert!(ladder_layer_indices(20, 21).is_none());
        assert!(ladder_layer_indices(20, 20).is_some());
        assert!(ladder_layer_indices(20, 2).is_some());
    }
}

//! Cooperative cancellation (PRD §31).
//!
//! A build checks the flag at stage boundaries and before spawning each new
//! LLM request; in-flight requests finish naturally (they are already bounded
//! by `max_concurrency`). Cancellation is refused once the pipeline enters the
//! publish critical section — that section must complete or recover via the
//! publish journal, never abort mid-way.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Shared cooperative-cancel flag. `Default` is "not cancelled".
#[derive(Debug, Clone, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub fn new() -> Self {
        Self::default()
    }

    /// Signals every holder to stop starting new work.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// Returns `Err(WikiError::Cancelled)` when the flag is set.
    pub fn check(&self) -> crate::error::Result<()> {
        if self.is_cancelled() {
            Err(crate::error::WikiError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_starts_clear_and_flips_for_all_clones() {
        let flag = CancelFlag::new();
        assert!(!flag.is_cancelled());
        assert!(flag.check().is_ok());

        let clone = flag.clone();
        flag.cancel();
        assert!(clone.is_cancelled());
        assert!(flag.check().is_err());
    }
}

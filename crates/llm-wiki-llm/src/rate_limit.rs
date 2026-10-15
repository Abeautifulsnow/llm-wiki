//! Outbound request-rate coordination (PRD §25/§32, LW-002).
//!
//! The bounded backoff in [`crate::transport`] REACTS to a 429; this limiter
//! avoids provoking one. It is shared by every request a provider instance
//! makes — one limiter per endpoint — so concurrent embedding batches pace
//! themselves against a single budget instead of each independently bursting
//! at a provider that can only take `requests_per_minute` of them.
//!
//! Scope: this coordinates requests from ONE process. Cross-process budgeting
//! (a second `llm-wiki embed` against the same provider) is out of reach for a
//! local limiter and stays the operator's concern.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;

/// Minimum-spacing limiter: consecutive request STARTS are kept at least
/// `60s / requests_per_minute` apart. Spacing (rather than a token bucket) is
/// deliberate — it protects a provider from the burst a bucket would allow,
/// and an idle period does not accumulate credit to spend later.
pub struct RateLimiter {
    min_interval: Duration,
    /// Earliest instant the next request may start; reserved under the lock.
    next_slot: Mutex<Instant>,
}

impl RateLimiter {
    /// `requests_per_minute == 0` disables pacing (`None`) — the historical
    /// behavior and the safe default for a low-quota or local endpoint.
    pub fn from_requests_per_minute(requests_per_minute: u32) -> Option<Arc<Self>> {
        (requests_per_minute > 0).then(|| {
            Arc::new(Self {
                min_interval: Self::interval(requests_per_minute),
                next_slot: Mutex::new(Instant::now()),
            })
        })
    }

    fn interval(requests_per_minute: u32) -> Duration {
        Duration::from_secs_f64(60.0 / f64::from(requests_per_minute.max(1)))
    }

    /// Waits for this caller's slot. Every request — retries included —
    /// reserves its own slot, so the observed start rate stays inside the
    /// budget however many tasks are in flight.
    pub async fn acquire(&self) {
        let wait = {
            let mut next_slot = self.next_slot.lock().await;
            let (wait, reserved) = reserve(*next_slot, Instant::now(), self.min_interval);
            *next_slot = reserved;
            wait
        };
        if !wait.is_zero() {
            tokio::time::sleep(wait).await;
        }
    }
}

/// Reserves the next slot: returns how long the caller must wait and the new
/// `next_slot`. Split out from [`RateLimiter::acquire`] so the scheduling
/// arithmetic is testable without a clock or a runtime.
///
/// An idle gap is NOT banked: when `now` is already past `next_slot` the slot
/// starts now, so a quiet period cannot be spent as a later burst.
fn reserve(next_slot: Instant, now: Instant, min_interval: Duration) -> (Duration, Instant) {
    let slot = next_slot.max(now);
    (slot.saturating_duration_since(now), slot + min_interval)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_requests_per_minute_builds_no_limiter() {
        assert!(RateLimiter::from_requests_per_minute(0).is_none());
        assert!(RateLimiter::from_requests_per_minute(1).is_some());
    }

    #[test]
    fn interval_is_one_minute_over_the_budget() {
        assert_eq!(RateLimiter::interval(120), Duration::from_millis(500));
        assert_eq!(RateLimiter::interval(60), Duration::from_secs(1));
        assert_eq!(RateLimiter::interval(6000), Duration::from_millis(10));
    }

    #[test]
    fn reserve_spaces_slots_and_does_not_bank_idle_time() {
        let interval = Duration::from_millis(500);
        let start = Instant::now();

        // First caller: free slot, and it books the next one.
        let (wait, next_slot) = reserve(start, start, interval);
        assert_eq!(wait, Duration::ZERO);
        assert_eq!(next_slot, start + interval);

        // Immediate second caller waits the full interval.
        let (wait, next_slot) = reserve(next_slot, start + Duration::from_millis(10), interval);
        assert_eq!(wait, Duration::from_millis(490));
        assert_eq!(next_slot, start + 2 * interval);

        // After a long idle gap the slot starts NOW — the gap is not banked
        // into a burst.
        let idle_now = start + Duration::from_secs(60);
        let (wait, next_slot) = reserve(start + interval, idle_now, interval);
        assert_eq!(wait, Duration::ZERO);
        assert_eq!(next_slot, idle_now + interval);
    }

    #[tokio::test]
    async fn acquire_paces_two_consecutive_callers() {
        // 6000/min = 10ms spacing; coarse enough to assert on real time.
        let limiter = RateLimiter::from_requests_per_minute(6000).unwrap();
        let started = Instant::now();
        limiter.acquire().await;
        limiter.acquire().await;
        assert!(
            started.elapsed() >= Duration::from_millis(8),
            "second acquire must wait out the interval, elapsed {:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn one_limiter_serializes_concurrent_batches() {
        // 600/min = 100ms spacing: three concurrent callers cannot all start
        // at once, so the third finishes no earlier than 2 intervals.
        let limiter = RateLimiter::from_requests_per_minute(600).unwrap();
        let started = Instant::now();
        let mut tasks = tokio::task::JoinSet::new();
        for _ in 0..3 {
            let limiter = limiter.clone();
            tasks.spawn(async move { limiter.acquire().await });
        }
        while tasks.join_next().await.is_some() {}
        assert!(
            started.elapsed() >= Duration::from_millis(180),
            "3 callers at 100ms spacing need >= 2 intervals, elapsed {:?}",
            started.elapsed()
        );
    }
}

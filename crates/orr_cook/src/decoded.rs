//! Decoded data that several cook jobs read, kept in memory up to a byte limit.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use orr_pack::Hash32;

/// A content hash and what distinguishes the values decoded from that content.
type Key = (Hash32, u32);

/// Values by key, the least recently used dropped once the held values pass a byte limit. A
/// job asking for a value another job is decoding waits for it, so decoding must not wait on
/// a rayon pool thread that could be asking for the same value.
pub(crate) struct Decoded<V> {
    limit: u64,
    size: fn(&V) -> u64,
    state: Mutex<State<V>>,
}

struct State<V> {
    slots: HashMap<Key, Slot<V>>,
    clock: u64,
}

impl<V> State<V> {
    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn held(&self) -> u64 {
        self.slots.values().map(|s| s.bytes).sum()
    }
}

struct Slot<V> {
    value: Arc<Mutex<Option<Arc<V>>>>,
    /// Size of the value; 0 until it is decoded.
    bytes: u64,
    used: u64,
}

impl<V> Decoded<V> {
    pub(crate) fn new(limit: u64, size: fn(&V) -> u64) -> Self {
        Self {
            limit,
            size,
            state: Mutex::new(State {
                slots: HashMap::new(),
                clock: 0,
            }),
        }
    }

    /// The value of `key`, from `decode` when it is not held. A value over the limit is
    /// shared only with the jobs already waiting for it.
    pub(crate) fn get<E>(
        &self,
        key: Key,
        decode: impl FnOnce() -> Result<V, E>,
    ) -> Result<Arc<V>, E> {
        let value = {
            let mut s = self.state.lock().expect("decoded lock");
            let used = s.tick();
            let slot = s.slots.entry(key).or_insert_with(|| Slot {
                value: Arc::default(),
                bytes: 0,
                used,
            });
            slot.used = used;
            Arc::clone(&slot.value)
        };
        // A decode that panicked left no value, and its own panic reports it.
        let mut value = value.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(v) = &*value {
            return Ok(Arc::clone(v));
        }
        let v = Arc::new(decode()?);
        *value = Some(Arc::clone(&v));
        self.hold(key, (self.size)(&v));
        Ok(v)
    }

    /// Counts a new value of `bytes`, dropping the least recently used values over the limit.
    fn hold(&self, key: Key, bytes: u64) {
        let mut s = self.state.lock().expect("decoded lock");
        if bytes > self.limit {
            s.slots.remove(&key);
            return;
        }
        let used = s.tick();
        let slot = s.slots.get_mut(&key).expect("a slot being decoded stays");
        slot.bytes = bytes;
        slot.used = used;
        while s.held() > self.limit {
            let oldest = s
                .slots
                .iter()
                .filter(|(_, slot)| slot.bytes > 0)
                .min_by_key(|(_, slot)| slot.used)
                .map(|(k, _)| *k)
                .expect("held bytes are in slots");
            s.slots.remove(&oldest);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    fn key(k: u32) -> Key {
        (Hash32::of(&[]), k)
    }

    /// Whether `k` is held: asking for it does not decode.
    fn held(c: &Decoded<u64>, k: u32) -> bool {
        let decoded = AtomicU32::new(0);
        let v = c.get::<()>(key(k), || {
            decoded.fetch_add(1, Ordering::Relaxed);
            Ok(0)
        });
        v.is_ok() && decoded.load(Ordering::Relaxed) == 0
    }

    #[test]
    fn drops_the_least_recently_used_past_the_limit() {
        let c = Decoded::new(10, |&bytes: &u64| bytes);
        let put = |k, bytes| c.get::<()>(key(k), || Ok(bytes)).unwrap();
        put(0, 4);
        put(1, 4);
        // Reading 0 makes 1 the oldest.
        assert_eq!(*put(0, 99), 4);
        put(2, 4);
        assert_eq!(c.state.lock().unwrap().held(), 8);
        assert!(!c.state.lock().unwrap().slots.contains_key(&key(1)));
        assert!(held(&c, 0) && held(&c, 2));
    }

    #[test]
    fn a_value_over_the_limit_is_not_held_and_drops_nothing() {
        let c = Decoded::new(10, |&bytes: &u64| bytes);
        c.get::<()>(key(0), || Ok(4)).unwrap();
        assert_eq!(*c.get::<()>(key(1), || Ok(11)).unwrap(), 11);
        assert_eq!(c.state.lock().unwrap().held(), 4);
        assert!(!c.state.lock().unwrap().slots.contains_key(&key(1)));
        assert!(held(&c, 0));
    }

    #[test]
    fn a_failed_decode_leaves_the_next_job_to_try_again() {
        let c = Decoded::new(10, |&bytes: &u64| bytes);
        assert_eq!(c.get(key(0), || Err("no")).unwrap_err(), "no");
        assert_eq!(*c.get::<()>(key(0), || Ok(1)).unwrap(), 1);
    }

    #[test]
    fn jobs_asking_at_once_share_one_decode() {
        // A limit of 0 also holds for jobs already waiting when the value is dropped.
        for limit in [0, 100] {
            let c = Decoded::new(limit, |&bytes: &u64| bytes);
            let decodes = AtomicU32::new(0);
            // Every job and the slot hold the value's cell before the decode finishes.
            let holders = || {
                let s = c.state.lock().unwrap();
                Arc::strong_count(&s.slots[&key(0)].value)
            };
            let decode = || {
                decodes.fetch_add(1, Ordering::Relaxed);
                while holders() < 9 {
                    std::thread::yield_now();
                }
                Ok::<_, ()>(5)
            };
            let job = || c.get(key(0), decode).unwrap();
            let values: Vec<Arc<u64>> = std::thread::scope(|s| {
                let jobs: Vec<_> = Vec::from_fn(8, |_| s.spawn(job));
                jobs.into_iter().map(|j| j.join().unwrap()).collect()
            });
            assert_eq!(decodes.load(Ordering::Relaxed), 1, "limit {limit}");
            assert!(values.iter().all(|v| Arc::ptr_eq(v, &values[0])));
        }
    }
}

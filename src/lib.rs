//! # ternary-event-pool
//!
//! Event pool for GPU kernel completion tracking.
//!
//! This crate provides an event pool system for managing GPU events,
//! tracking their states, maintaining ordered timelines, and supporting
//! blocking waits for event completion. Designed for ternary GPU compute
//! pipelines where fine-grained kernel completion tracking is needed.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex, Condvar};
use std::time::{Duration, Instant};

/// Unique identifier for an event.
pub type EventId = u64;

/// State of a GPU event in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventState {
    /// Event is available in the pool, not yet in use.
    Free,
    /// Event has been acquired and is pending (kernel not yet launched or recording).
    Pending,
    /// Event is currently being recorded by a GPU kernel.
    Recording,
    /// Event has been recorded; the associated kernel has completed.
    Recorded,
}

impl fmt::Display for EventState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EventState::Free => write!(f, "Free"),
            EventState::Pending => write!(f, "Pending"),
            EventState::Recording => write!(f, "Recording"),
            EventState::Recorded => write!(f, "Recorded"),
        }
    }
}

/// A single GPU event with metadata.
#[derive(Debug, Clone)]
pub struct Event {
    /// Unique event identifier.
    pub id: EventId,
    /// Current state.
    pub state: EventState,
    /// Timestamp when the event was created.
    pub created_at: Instant,
    /// Timestamp when the event was recorded (kernel completed).
    pub recorded_at: Option<Instant>,
    /// Optional label for debugging.
    pub label: Option<String>,
}

impl Event {
    /// Create a new event with the given ID.
    pub fn new(id: EventId) -> Self {
        Self {
            id,
            state: EventState::Free,
            created_at: Instant::now(),
            recorded_at: None,
            label: None,
        }
    }

    /// Create a new event with a label.
    pub fn with_label(id: EventId, label: impl Into<String>) -> Self {
        Self {
            label: Some(label.into()),
            ..Self::new(id)
        }
    }

    /// Mark the event as pending (acquired from pool).
    pub fn mark_pending(&mut self) -> Result<(), EventError> {
        if self.state != EventState::Free {
            return Err(EventError::InvalidTransition {
                from: self.state,
                to: EventState::Pending,
                event_id: self.id,
            });
        }
        self.state = EventState::Pending;
        Ok(())
    }

    /// Mark the event as recording.
    pub fn mark_recording(&mut self) -> Result<(), EventError> {
        if self.state != EventState::Pending {
            return Err(EventError::InvalidTransition {
                from: self.state,
                to: EventState::Recording,
                event_id: self.id,
            });
        }
        self.state = EventState::Recording;
        Ok(())
    }

    /// Mark the event as recorded (kernel completed).
    pub fn mark_recorded(&mut self) -> Result<(), EventError> {
        if self.state != EventState::Recording {
            return Err(EventError::InvalidTransition {
                from: self.state,
                to: EventState::Recorded,
                event_id: self.id,
            });
        }
        self.state = EventState::Recorded;
        self.recorded_at = Some(Instant::now());
        Ok(())
    }

    /// Reset the event back to free state for reuse.
    pub fn reset(&mut self) {
        self.state = EventState::Free;
        self.recorded_at = None;
        self.created_at = Instant::now();
    }

    /// Whether the event is complete (recorded).
    pub fn is_complete(&self) -> bool {
        self.state == EventState::Recorded
    }

    /// Elapsed time since event creation.
    pub fn elapsed(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Duration the kernel took (from creation to recorded).
    pub fn kernel_duration(&self) -> Option<Duration> {
        self.recorded_at.map(|r| r.duration_since(self.created_at))
    }
}

/// Errors that can occur during event operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventError {
    /// Invalid state transition attempted.
    InvalidTransition {
        from: EventState,
        to: EventState,
        event_id: EventId,
    },
    /// No events available in the pool.
    PoolExhausted,
    /// Event not found.
    NotFound(EventId),
    /// Timeout waiting for events.
    Timeout,
}

impl fmt::Display for EventError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EventError::InvalidTransition { from, to, event_id } => {
                write!(f, "invalid transition from {} to {} for event {}", from, to, event_id)
            }
            EventError::PoolExhausted => write!(f, "event pool exhausted"),
            EventError::NotFound(id) => write!(f, "event {} not found", id),
            EventError::Timeout => write!(f, "timeout waiting for events"),
        }
    }
}

impl std::error::Error for EventError {}

/// An entry in the event timeline.
#[derive(Debug, Clone)]
pub struct TimelineEntry {
    /// The event.
    pub event: Event,
    /// Sequence number (monotonically increasing).
    pub seq: u64,
}

/// Ordered timeline of event history.
#[derive(Debug, Clone)]
pub struct EventTimeline {
    entries: VecDeque<TimelineEntry>,
    next_seq: u64,
    max_entries: usize,
}

impl EventTimeline {
    /// Create a new timeline with optional max capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_entries.min(1024)),
            next_seq: 0,
            max_entries,
        }
    }

    /// Record an event in the timeline.
    pub fn record(&mut self, event: &Event) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;

        let entry = TimelineEntry {
            event: event.clone(),
            seq,
        };

        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
        seq
    }

    /// Get all entries in order.
    pub fn entries(&self) -> &VecDeque<TimelineEntry> {
        &self.entries
    }

    /// Number of entries in the timeline.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the timeline is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Verify timeline entries are in order.
    pub fn verify_ordering(&self) -> bool {
        self.entries
            .iter()
            .zip(self.entries.iter().skip(1))
            .all(|(a, b)| a.seq < b.seq)
    }

    /// Get entries by state.
    pub fn by_state(&self, state: EventState) -> Vec<&TimelineEntry> {
        self.entries.iter().filter(|e| e.event.state == state).collect()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Internal shared state for the event pool.
struct PoolInner {
    /// All events in the pool.
    events: Vec<Event>,
    /// Index of free events.
    free_indices: Vec<usize>,
    /// Next event ID to allocate.
    next_id: EventId,
}

/// A pre-allocated pool of GPU events.
pub struct EventPool {
    inner: Arc<Mutex<PoolInner>>,
    timeline: Arc<Mutex<EventTimeline>>,
    /// Condvar for signaling event completion.
    completion_signal: Arc<(Mutex<Vec<EventId>>, Condvar)>,
}

impl EventPool {
    /// Create a new event pool with `capacity` pre-allocated events.
    pub fn new(capacity: usize) -> Self {
        let mut events = Vec::with_capacity(capacity);
        let mut free_indices = Vec::with_capacity(capacity);
        for i in 0..capacity {
            events.push(Event::new(i as u64));
            free_indices.push(i);
        }

        Self {
            inner: Arc::new(Mutex::new(PoolInner {
                events,
                free_indices,
                next_id: capacity as u64,
            })),
            timeline: Arc::new(Mutex::new(EventTimeline::new(1000))),
            completion_signal: Arc::new((Mutex::new(Vec::new()), Condvar::new())),
        }
    }

    /// Total pool capacity.
    pub fn capacity(&self) -> usize {
        self.inner.lock().unwrap().events.len()
    }

    /// Number of available (free) events.
    pub fn available(&self) -> usize {
        self.inner.lock().unwrap().free_indices.len()
    }

    /// Acquire an event from the pool.
    pub fn acquire(&self) -> Result<Event, EventError> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(idx) = inner.free_indices.pop() {
            inner.events[idx].mark_pending()?;
            let event = inner.events[idx].clone();
            // Record in timeline
            let mut timeline = self.timeline.lock().unwrap();
            timeline.record(&event);
            Ok(event)
        } else {
            Err(EventError::PoolExhausted)
        }
    }

    /// Acquire an event with a label.
    pub fn acquire_labeled(&self, label: impl Into<String>) -> Result<Event, EventError> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(idx) = inner.free_indices.pop() {
            inner.events[idx].label = Some(label.into());
            inner.events[idx].mark_pending()?;
            let event = inner.events[idx].clone();
            let mut timeline = self.timeline.lock().unwrap();
            timeline.record(&event);
            Ok(event)
        } else {
            Err(EventError::PoolExhausted)
        }
    }

    /// Mark an event as recording.
    pub fn begin_recording(&self, event_id: EventId) -> Result<(), EventError> {
        let mut inner = self.inner.lock().unwrap();
        let event = inner.events.iter_mut()
            .find(|e| e.id == event_id)
            .ok_or(EventError::NotFound(event_id))?;
        event.mark_recording()
    }

    /// Mark an event as recorded (kernel completed).
    pub fn complete(&self, event_id: EventId) -> Result<(), EventError> {
        let mut inner = self.inner.lock().unwrap();
        let event = inner.events.iter_mut()
            .find(|e| e.id == event_id)
            .ok_or(EventError::NotFound(event_id))?;
        event.mark_recorded()?;

        // Signal completion
        {
            let (lock, cvar) = &*self.completion_signal;
            let mut completed = lock.lock().unwrap();
            completed.push(event_id);
            cvar.notify_all();
        }

        // Update timeline
        let event_clone = event.clone();
        drop(inner);
        let mut timeline = self.timeline.lock().unwrap();
        timeline.record(&event_clone);

        Ok(())
    }

    /// Release an event back to the pool for reuse.
    pub fn release(&self, event_id: EventId) -> Result<(), EventError> {
        let mut inner = self.inner.lock().unwrap();
        let idx = inner.events.iter().position(|e| e.id == event_id)
            .ok_or(EventError::NotFound(event_id))?;

        inner.events[idx].reset();
        inner.free_indices.push(idx);
        Ok(())
    }

    /// Get the timeline.
    pub fn timeline(&self) -> EventTimeline {
        self.timeline.lock().unwrap().clone()
    }

    /// Get event state by ID.
    pub fn get_state(&self, event_id: EventId) -> Option<EventState> {
        let inner = self.inner.lock().unwrap();
        inner.events.iter().find(|e| e.id == event_id).map(|e| e.state)
    }

    /// Expand the pool by adding more events.
    pub fn expand(&self, additional: usize) {
        let mut inner = self.inner.lock().unwrap();
        let start = inner.next_id;
        for i in 0..additional {
            let id = start + i as u64;
            let idx = inner.events.len();
            inner.events.push(Event::new(id));
            inner.free_indices.push(idx);
        }
        inner.next_id = start + additional as u64;
    }
}

/// Wait for a set of events to complete, blocking the calling thread.
pub fn wait_for_events(pool: &EventPool, event_ids: &[EventId], timeout: Option<Duration>) -> Result<(), EventError> {
    let start = Instant::now();
    let (lock, cvar) = &*pool.completion_signal;

    let remaining: std::collections::HashSet<EventId> = event_ids.iter().copied().collect();

    let mut completed = lock.lock().unwrap();
    loop {
        // Check which events are done
        let inner = pool.inner.lock().unwrap();
        let all_done = event_ids.iter().all(|id| {
            inner.events.iter().any(|e| e.id == *id && e.state == EventState::Recorded)
        });
        drop(inner);

        if all_done {
            // Clear completed events that we were waiting for
            completed.retain(|id| !event_ids.contains(id));
            return Ok(());
        }

        if let Some(t) = timeout {
            if start.elapsed() >= t {
                return Err(EventError::Timeout);
            }
            let remaining_timeout = t.checked_sub(start.elapsed()).unwrap_or(Duration::ZERO);
            let result = cvar.wait_timeout(completed, remaining_timeout).unwrap();
            completed = result.0;
        } else {
            completed = cvar.wait(completed).unwrap();
        }
    }
}

/// Wait for a single event to complete.
pub fn wait_for_event(pool: &EventPool, event_id: EventId, timeout: Option<Duration>) -> Result<(), EventError> {
    wait_for_events(pool, &[event_id], timeout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn test_pool_allocates_events() {
        let pool = EventPool::new(8);
        assert_eq!(pool.capacity(), 8);
        assert_eq!(pool.available(), 8);

        let e1 = pool.acquire().unwrap();
        assert_eq!(e1.state, EventState::Pending);
        assert_eq!(pool.available(), 7);

        let e2 = pool.acquire().unwrap();
        assert_eq!(pool.available(), 6);
    }

    #[test]
    fn test_pool_exhausted() {
        let pool = EventPool::new(2);
        pool.acquire().unwrap();
        pool.acquire().unwrap();
        let result = pool.acquire();
        assert!(matches!(result, Err(EventError::PoolExhausted)));
    }

    #[test]
    fn test_state_transitions() {
        let pool = EventPool::new(4);
        let e = pool.acquire().unwrap();
        assert_eq!(e.state, EventState::Pending);

        // Pending → Recording
        pool.begin_recording(e.id).unwrap();
        assert_eq!(pool.get_state(e.id), Some(EventState::Recording));

        // Recording → Recorded
        pool.complete(e.id).unwrap();
        assert_eq!(pool.get_state(e.id), Some(EventState::Recorded));
    }

    #[test]
    fn test_invalid_transitions() {
        // Free → Recorded is invalid
        let mut event = Event::new(42);
        assert!(event.mark_recorded().is_err());

        // Pending → Recorded is invalid
        let pool = EventPool::new(4);
        let e = pool.acquire().unwrap();
        assert!(pool.complete(e.id).is_err()); // Pending → Recorded invalid
    }

    #[test]
    fn test_double_recording_fails() {
        let pool = EventPool::new(4);
        let e = pool.acquire().unwrap();
        pool.begin_recording(e.id).unwrap();
        pool.complete(e.id).unwrap();
        // Try to complete again
        assert!(pool.complete(e.id).is_err());
    }

    #[test]
    fn test_timeline_ordering() {
        let pool = EventPool::new(4);
        let e1 = pool.acquire().unwrap();
        let e2 = pool.acquire().unwrap();
        let e3 = pool.acquire().unwrap();

        let timeline = pool.timeline();
        assert!(timeline.len() >= 3);
        assert!(timeline.verify_ordering());

        // Verify sequence numbers are increasing
        let entries: Vec<_> = timeline.entries().iter().collect();
        assert!(entries[0].seq < entries[1].seq);
        assert!(entries[1].seq < entries[2].seq);
    }

    #[test]
    fn test_timeline_max_entries() {
        let pool = EventPool::new(100);
        for i in 0..100 {
            pool.acquire_labeled(format!("event-{}", i)).unwrap();
        }
        // Also complete some
        for i in 0..50 {
            pool.begin_recording(i).unwrap();
            pool.complete(i).unwrap();
        }

        let timeline = pool.timeline();
        // Should have at most 1000 entries (our max)
        assert!(timeline.len() <= 1000);
    }

    #[test]
    fn test_event_reuse() {
        let pool = EventPool::new(2);
        let e1 = pool.acquire().unwrap();
        let _e2 = pool.acquire().unwrap();
        pool.begin_recording(e1.id).unwrap();
        pool.complete(e1.id).unwrap();
        // Pool is now exhausted (0 available), e1 is Recorded, e2 is Pending
        assert_eq!(pool.available(), 0);

        // Release e1 for reuse
        pool.release(e1.id).unwrap();
        assert_eq!(pool.available(), 1);

        // Acquire again
        let e1_again = pool.acquire().unwrap();
        assert_eq!(e1_again.state, EventState::Pending);
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn test_event_labeled() {
        let pool = EventPool::new(4);
        let e = pool.acquire_labeled("kernel-42").unwrap();
        assert_eq!(e.label.as_deref(), Some("kernel-42"));
    }

    #[test]
    fn test_event_is_complete() {
        let mut event = Event::new(1);
        assert!(!event.is_complete());
        event.mark_pending().unwrap();
        assert!(!event.is_complete());
        event.mark_recording().unwrap();
        assert!(!event.is_complete());
        event.mark_recorded().unwrap();
        assert!(event.is_complete());
    }

    #[test]
    fn test_event_reset() {
        let mut event = Event::new(1);
        event.mark_pending().unwrap();
        event.mark_recording().unwrap();
        event.mark_recorded().unwrap();
        assert!(event.recorded_at.is_some());

        event.reset();
        assert_eq!(event.state, EventState::Free);
        assert!(event.recorded_at.is_none());
    }

    #[test]
    fn test_event_not_found() {
        let pool = EventPool::new(4);
        let result = pool.begin_recording(999);
        assert!(matches!(result, Err(EventError::NotFound(999))));
    }

    #[test]
    fn test_wait_blocks_until_complete() {
        let pool = Arc::new(EventPool::new(4));
        let e = pool.acquire().unwrap();
        pool.begin_recording(e.id).unwrap();

        let pool_clone = Arc::clone(&pool);
        let event_id = e.id;
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            pool_clone.complete(event_id).unwrap();
        });

        // This should block until the event completes
        let result = wait_for_event(&pool, e.id, Some(Duration::from_secs(5)));
        assert!(result.is_ok());
        handle.join().unwrap();
    }

    #[test]
    fn test_wait_timeout() {
        let pool = EventPool::new(4);
        let e = pool.acquire().unwrap();
        pool.begin_recording(e.id).unwrap();

        // Don't complete the event — should timeout
        let result = wait_for_event(&pool, e.id, Some(Duration::from_millis(50)));
        assert!(matches!(result, Err(EventError::Timeout)));
    }

    #[test]
    fn test_wait_multiple_events() {
        let pool = Arc::new(EventPool::new(4));
        let e1 = pool.acquire().unwrap();
        let e2 = pool.acquire().unwrap();
        pool.begin_recording(e1.id).unwrap();
        pool.begin_recording(e2.id).unwrap();

        let pool_clone = Arc::clone(&pool);
        let id1 = e1.id;
        let id2 = e2.id;
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(30));
            pool_clone.complete(id1).unwrap();
            thread::sleep(Duration::from_millis(30));
            pool_clone.complete(id2).unwrap();
        });

        let result = wait_for_events(&pool, &[e1.id, e2.id], Some(Duration::from_secs(5)));
        assert!(result.is_ok());
        handle.join().unwrap();
    }

    #[test]
    fn test_pool_expand() {
        let pool = EventPool::new(2);
        assert_eq!(pool.capacity(), 2);

        pool.expand(3);
        assert_eq!(pool.capacity(), 5);
        assert_eq!(pool.available(), 5);
    }

    #[test]
    fn test_event_kernel_duration() {
        let mut event = Event::new(1);
        assert!(event.kernel_duration().is_none());

        event.mark_pending().unwrap();
        event.mark_recording().unwrap();
        event.mark_recorded().unwrap();
        assert!(event.kernel_duration().is_some());
    }

    #[test]
    fn test_event_error_display() {
        let err = EventError::PoolExhausted;
        assert_eq!(format!("{}", err), "event pool exhausted");

        let err = EventError::NotFound(42);
        assert_eq!(format!("{}", err), "event 42 not found");
    }

    #[test]
    fn test_timeline_by_state() {
        let pool = EventPool::new(4);
        let e1 = pool.acquire().unwrap();
        let _e2 = pool.acquire().unwrap();

        pool.begin_recording(e1.id).unwrap();
        pool.complete(e1.id).unwrap();

        let timeline = pool.timeline();
        let recorded = timeline.by_state(EventState::Recorded);
        assert!(!recorded.is_empty());
    }

    #[test]
    fn test_timeline_clear() {
        let pool = EventPool::new(4);
        pool.acquire().unwrap();
        pool.acquire().unwrap();

        let mut timeline = pool.timeline();
        assert!(!timeline.is_empty());
        timeline.clear();
        assert!(timeline.is_empty());
    }
}

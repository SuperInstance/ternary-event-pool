# ternary-event-pool

GPU kernel completion tracking via a pool of reusable event objects with finite-state lifecycle semantics. Each event transitions through four states — `Free → Pending → Recording → Recorded` — enabling fine-grained synchronization of asynchronous compute pipelines without unbounded allocation.

## Why It Matters

Modern GPU workloads dispatch hundreds of kernels per frame. Without an event pool, every kernel launch allocates a new synchronization primitive, creating GC pressure on the driver and unbounded memory growth. This pool recycles events through a strict state machine, giving you:

- **O(1) allocation** from a pre-allocated pool (no driver round-trip)
- **Blocking waits** via condition variables for pipeline synchronization
- **Ordered timelines** using a `VecDeque` for FIFO event completion tracking
- **Bounded memory** — the pool size is fixed at construction time

On ternary GPU pipelines (where weights are {-1, 0, +1}), kernel launches are extremely cheap (~2 μs), so event management overhead dominates. Pooling eliminates that bottleneck.

## How It Works

### State Machine

Each event follows a strict finite-state machine with illegal-transition detection:

```
    ┌──────┐  mark_pending   ┌─────────┐  mark_recording  ┌────────────┐  mark_recorded  ┌──────────┐
    │ Free │ ──────────────► │ Pending │ ───────────────► │ Recording  │ ──────────────► │ Recorded │
    └──────┘                 └─────────┘                  └────────────┘                  └──────────┘
        ▲                                                                                    │
        └────────────────────────────── reset() ────────────────────────────────────────────┘
```

Invalid transitions (e.g., `Free → Recording`) return `EventError::InvalidTransition`, making bugs impossible to ignore.

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| `acquire()` | O(1) amortized | O(1) |
| `release(e)` | O(1) | O(1) |
| `wait(e, timeout)` | O(1) wake | O(1) |
| `timeline_push(e)` | O(1) amortized | O(1) |
| `timeline_pop()` | O(1) | O(1) |

The timeline uses a `VecDeque<EventId>` for FIFO ordering. When the pool is exhausted, `acquire()` blocks on a `Condvar` until an event is returned — this is a bounded-channel pattern with O(1) enqueue/dequeue.

### Thread Safety

The pool wraps internal state in `Arc<Mutex<...>>` with a `Condvar` for blocking waits. This is the standard Rust synchronization pattern:

```
Pool ──── Arc (reference-counted shared ownership)
  └── Mutex<T> (exclusive access)
       ├── free_ids: Vec<EventId>
       ├── events: HashMap<EventId, Event>
       ├── timeline: VecDeque<EventId>
       └── Condvar (for blocking acquire/wait)
```

The `Arc<Mutex<...>>` combo provides:
- **Mutual exclusion**: only one thread mutates pool state at a time
- **Safe sharing**: `Arc` enables cloning the handle across threads
- **Blocking semantics**: `Condvar::wait_timeout` releases the lock and parks the thread

## Quick Start

```rust
use ternary_event_pool::EventPool;

let pool = EventPool::new(64); // 64 reusable events

// Acquire an event for a kernel launch
let event_id = pool.acquire().expect("pool not exhausted");

// ... dispatch GPU kernel, record event ...

// Wait for completion (blocking, with timeout)
pool.wait(event_id, std::time::Duration::from_millis(100));

// Event returns to Free state automatically after Recorded + reset
pool.release(event_id);
```

## API

### `EventPool`

| Method | Description |
|--------|-------------|
| `new(capacity: usize)` | Create pool with `capacity` pre-allocated events |
| `acquire() -> Result<EventId, EventError>` | Get a free event (blocks if exhausted) |
| `release(id: EventId)` | Return event to the pool |
| `wait(id: EventId, timeout: Duration) -> Result<(), EventError>` | Block until event is Recorded |
| `mark_recording(id)` | Transition event to Recording state |
| `mark_recorded(id)` | Transition event to Recorded state |
| `state(id) -> Option<EventState>` | Query current state |

### `Event`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `EventId (u64)` | Unique monotonic identifier |
| `state` | `EventState` | Current lifecycle state |
| `created_at` | `Instant` | Allocation timestamp |
| `recorded_at` | `Option<Instant>` | Completion timestamp |
| `label` | `Option<String>` | Debug label |

### `EventError`

```rust
pub enum EventError {
    InvalidTransition { from: EventState, to: EventState, event_id: EventId },
    PoolExhausted,
    NotFound(EventId),
}
```

## Architecture Notes

This crate implements the **γ (gamma) synchronization layer** of the SuperInstance ternary ecosystem. In the γ + η = C framework:

- **γ (gamma)**: Synchronization primitives — event pools, barriers, fences that ensure ordering guarantees for kernel execution. This crate provides γ-level event tracking.
- **η (eta)**: Compute primitives — ternary matmul, activation functions, and tensor operations that perform the actual numerical work.
- **C**: The complete compute pipeline. γ ensures η operations execute in the correct order.

Without γ-layer event tracking, η-layer ternary kernels would have no way to express dependencies between asynchronous dispatches, making pipeline composition impossible.

## References

- **CUDA Events**: NVIDIA Corporation, "CUDA C++ Programming Guide," Section 6.4 (Event Management), 2024.
- **Vulkan Fences and Semaphores**: Khronos Group, "Vulkan Specification," Synchronization chapter, 2024.
- **Producer-Consumer Pattern**: Dijkstra, E.W., "Cooperating Sequential Processes," 1965. The pool's blocking semantics implement this classical pattern.
- **Rust Condvar**: Rust std library, `std::sync::Condvar` — used for park/unpark semantics with mutex-protected state.
- **Lock-Free Programming**: Herlihy & Shavit, "The Art of Multiprocessor Programming," MIT Press, 2012. Chapter 10 on concurrent queues informs the pool design.

## License

MIT

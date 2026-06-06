# ternary-event-pool

Event pool for GPU kernel completion tracking.

## Overview

`ternary-event-pool` provides a pre-allocated pool of GPU events with lifecycle management, state tracking, ordered timeline history, and blocking wait primitives. Designed for ternary GPU compute pipelines where fine-grained kernel completion tracking and synchronization are essential.

## Features

- **EventPool** — Pre-allocate GPU events, acquire/release them with automatic lifecycle management, and expand the pool dynamically.
- **EventState** — Track event state transitions (Free → Pending → Recording → Recorded) with strict validation preventing illegal transitions.
- **EventTimeline** — Maintain an ordered, bounded history of all event operations with sequence numbers and state-based filtering.
- **wait_for_events** — Block the calling thread until one or more events complete, with optional timeout support using condition variables.
- **Event Reuse** — Release completed events back to the pool for recycling, reducing allocation overhead in long-running pipelines.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
ternary-event-pool = { git = "https://github.com/SuperInstance/ternary-event-pool" }
```

### Basic Example

```rust
use ternary_event_pool::*;

// Create a pool with 16 pre-allocated events
let pool = EventPool::new(16);

// Acquire an event for kernel tracking
let event = pool.acquire_labeled("kernel-42").unwrap();
println!("Event {} state: {}", event.id, event.state);

// Begin recording (kernel launch)
pool.begin_recording(event.id).unwrap();

// Complete the event (kernel finished)
pool.complete(event.id).unwrap();
println!("Kernel duration: {:?}", event.kernel_duration());

// Release back to pool for reuse
pool.release(event.id).unwrap();
```

### Waiting for Multiple Events

```rust
use ternary_event_pool::*;
use std::time::Duration;

let pool = EventPool::new(8);

let e1 = pool.acquire().unwrap();
let e2 = pool.acquire().unwrap();

pool.begin_recording(e1.id).unwrap();
pool.begin_recording(e2.id).unwrap();

// ... launch kernels ...

pool.complete(e1.id).unwrap();
pool.complete(e2.id).unwrap();

// Block until both complete (5-second timeout)
wait_for_events(&pool, &[e1.id, e2.id], Some(Duration::from_secs(5))).unwrap();
```

## Architecture

### Event Lifecycle

Events follow a strict state machine:

```
Free → Pending → Recording → Recorded
  ↑                              |
  └──────── reset() ─────────────┘
```

Each transition is validated — attempting an invalid transition returns `EventError::InvalidTransition`.

### EventPool

The pool pre-allocates events on creation. Events are acquired (moved to Pending), transitioned through Recording to Recorded, and then released back to Free for reuse. The pool tracks free indices for O(1) acquire/release. Dynamic expansion via `expand()` adds more events with monotonically increasing IDs.

### EventTimeline

Every state change is recorded in a bounded timeline (default 1000 entries, oldest evicted first). Each entry has a monotonically increasing sequence number. The timeline supports filtering by state and verifying ordering integrity.

### Blocking Wait

`wait_for_events` uses condition variables to efficiently block until all specified events reach the Recorded state. Supports optional timeouts. Thread-safe via `Arc<Mutex<>>` internally.

## Thread Safety

The `EventPool` is thread-safe — all internal state is protected by mutexes. The `wait_for_events` function uses condvars for efficient cross-thread signaling, making it suitable for producer-consumer patterns where one thread launches kernels and another waits for completion.

## License

MIT

// Request IDs are used once per trace. No GPU/model/session semantics.
module scheduler {
  val ids = 0.to(3)
  var phase: int -> int // fresh, queued, granted, running, cleaning, settled
  var queue: List[int]
  var cancelled: Set[int]
  var failed: Set[int]
  var releases: int
  var grants: int
  var capacity: int
  var queueLimit: int
  var event: str
  var request: int

  val active = ids.filter(i => Set(2, 3, 4).contains(phase.get(i)))
  val interactive = queue.select(i => i % 2 == 0)
  val best = if (interactive.length() > 0) interactive.head() else queue.head()

  action init = {
    nondet slots = Set(1, 2).oneOf()
    nondet bound = Set(1, 2).oneOf()
    all {
      phase' = ids.mapBy(_ => 0), queue' = [], cancelled' = Set(), failed' = Set(),
      releases' = 0, grants' = 0, capacity' = slots, queueLimit' = bound,
      event' = "init", request' = -1
    }
  }
  action arrive(i) = all {
    phase.get(i) == 0,
    if (active.size() < capacity) all {
      phase' = phase.set(i, 2), queue' = queue, grants' = grants + 1
    } else if (queue.length() < queueLimit) all {
      phase' = phase.set(i, 1), queue' = queue.append(i), grants' = grants
    } else all { phase' = phase.set(i, 5), queue' = queue, grants' = grants },
    cancelled' = cancelled, failed' = failed, releases' = releases,
    capacity' = capacity, queueLimit' = queueLimit, event' = "arrive", request' = i
  }
  action start(i) = all {
    phase.get(i) == 2,
    phase' = phase.set(i, 3), queue' = queue, cancelled' = cancelled,
    failed' = failed, releases' = releases, grants' = grants,
    capacity' = capacity, queueLimit' = queueLimit, event' = "start", request' = i
  }
  action cancelQueued(i) = all {
    phase.get(i) == 1,
    phase' = phase.set(i, 5), queue' = queue.select(j => j != i),
    cancelled' = cancelled.union(Set(i)), failed' = failed,
    releases' = releases, grants' = grants, capacity' = capacity,
    queueLimit' = queueLimit, event' = "cancelQueued", request' = i
  }
  action timeout(i) = all {
    phase.get(i) == 1,
    phase' = phase.set(i, 5), queue' = queue.select(j => j != i),
    cancelled' = cancelled, failed' = failed, releases' = releases,
    grants' = grants, capacity' = capacity, queueLimit' = queueLimit,
    event' = "timeout", request' = i
  }
  action cancelActive(i) = all {
    Set(3, 4).contains(phase.get(i)), not(cancelled.contains(i)),
    phase' = phase, queue' = queue, cancelled' = cancelled.union(Set(i)),
    failed' = failed, releases' = releases, grants' = grants,
    capacity' = capacity, queueLimit' = queueLimit, event' = "cancelActive", request' = i
  }
  action cleanup(i, failure) = all {
    phase.get(i) == 3,
    phase' = phase.set(i, 4), queue' = queue, cancelled' = cancelled,
    failed' = if (failure) failed.union(Set(i)) else failed,
    releases' = releases, grants' = grants, capacity' = capacity,
    queueLimit' = queueLimit, event' = if (failure) "fail" else "cleanup", request' = i
  }
  action release(i, beforeStart) = all {
    phase.get(i) == (if (beforeStart) 2 else 4),
    if (queue.length() > 0) all {
      phase' = phase.set(i, 5).set(best, 2),
      queue' = queue.select(j => j != best), grants' = grants + 1
    } else all { phase' = phase.set(i, 5), queue' = queue, grants' = grants },
    cancelled' = if (beforeStart) cancelled.union(Set(i)) else cancelled,
    failed' = failed, releases' = releases + 1, capacity' = capacity,
    queueLimit' = queueLimit, event' = if (beforeStart) "cancelGranted" else "release", request' = i
  }
  action step = {
    nondet i = ids.oneOf()
    any {
      arrive(i), start(i), cancelQueued(i), timeout(i), cancelActive(i),
      cleanup(i, false), cleanup(i, true), release(i, false), release(i, true)
    }
  }
  val safety = all {
    active.size() <= capacity,
    queue.length() <= queueLimit,
    queue.length() == queue.foldl(Set(), (s, i) => s.union(Set(i))).size(),
    queue.foldl(Set(), (s, i) => s.union(Set(i))) == ids.filter(i => phase.get(i) == 1),
    grants - releases == active.size(),
    releases >= 0,
    ids.forall(i => phase.get(i) >= 0 and phase.get(i) <= 5)
  }
}

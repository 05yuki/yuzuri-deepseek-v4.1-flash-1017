// Exact immutable FP8 engram row retrieval with a bounded direct-mapped RAM
// cache in front of O_DIRECT NVMe reads.
//
// Derived from adapter/row_store.cpp in
// https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000
// Copyright (c) 2026 0xSero, MIT License. Modifications for the KTransformers
// dual-EPYC host (kvcache-ai/ktransformers fork, 05yuki):
//   - never abort inside the callback: failures zero the row, bump an error
//     counter, and are surfaced by the Python side after stream sync;
//   - 64 KiB read staging so one pread covers a row straddling a 4 KiB edge;
//   - stats expose cache slot count and bytes so the budget is auditable;
//   - lock-free atomic batching, offset-sorted deduplicated prefill reads at
//     high queue depth, unified layer-ordered prewarm, and live-lookup priority
//     so background warming never contends with foreground lookups.
//
// No CUDA calls in the callback: suitable for cudaLaunchHostFunc graph nodes.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <thread>
#include <vector>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr uint64_t kDim = 256;       // fp8 e4m3 bytes per row
constexpr uint64_t kScales = 8;      // e8m0 bytes per row (block 32)
constexpr uint64_t kMaxRow = kDim + kScales;
constexpr size_t kPage = 4096;
constexpr int kLockShards = 256;

std::atomic<uint64_t> g_lookups{0};
std::atomic<uint64_t> g_stores{0};
std::atomic<int> g_active_lookups{0};

struct UniqueMiss {
  int64_t id;
  bool ok;
  uint8_t row[kMaxRow];
};

struct Store {
  int fd = -1;
  std::string path;
  uint64_t rows = 0, weight_offset = 0, scale_offset = 0, slots = 0;
  uint64_t wbytes = kDim, sbytes = kScales, row_bytes = kDim + kScales;
  uint64_t row_lo = 0, row_hi = 0;
  uint8_t *cache = nullptr;
  uint64_t *keys = nullptr;
  uint8_t *resident = nullptr;
  size_t resident_size = 0;
  uint8_t *scales_ram = nullptr;
  size_t scales_ram_size = 0;
  std::mutex locks[kLockShards];
  std::atomic<uint64_t> hits{0}, misses{0}, reads{0}, errors{0};
  std::atomic<uint64_t> calls{0}, lookup_ns{0}, miss_ns{0};
  std::atomic<uint64_t> warm_rows{0}, warm_ns{0};

  // Persistent low-latency thread pool for decode-sized miss batches (lock-free atomic fetch).
  struct IoPool {
    std::vector<std::thread> threads;
    std::mutex mu;
    std::condition_variable cv, done_cv;
    UniqueMiss *batch = nullptr;
    size_t count = 0;
    std::atomic<size_t> next{0};
    uint64_t gen = 0;
    size_t active = 0;
    bool stop = false;
  } io;
};

bool read_row(Store *s, uint64_t id, uint8_t *row);

void pin_io_thread() {
  const char *e = getenv("DSV41_ENGRAM_IO_CPUS");
  if (!e || !*e) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  for (const char *p = e; *p;) {
    char *end;
    long a = std::strtol(p, &end, 10), b = a;
    if (end == p) break;
    if (*end == '-') b = std::strtol(end + 1, &end, 10);
    for (long c = a; c <= b && c < CPU_SETSIZE; c++) CPU_SET(c, &set);
    p = *end == ',' ? end + 1 : end;
  }
  if (CPU_COUNT(&set)) pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

void io_worker(Store *s) {
  pin_io_thread();
  uint64_t seen = 0;
  for (;;) {
    UniqueMiss *batch = nullptr;
    size_t count = 0;
    {
      std::unique_lock<std::mutex> lk(s->io.mu);
      s->io.cv.wait(lk, [&] { return s->io.stop || s->io.gen != seen; });
      if (s->io.stop) return;
      seen = s->io.gen;
      batch = s->io.batch;
      count = s->io.count;
    }
    for (size_t i; (i = s->io.next.fetch_add(1, std::memory_order_relaxed)) < count;) {
      batch[i].ok = read_row(s, uint64_t(batch[i].id), batch[i].row);
    }
    std::lock_guard<std::mutex> lk(s->io.mu);
    if (--s->io.active == 0) s->io.done_cv.notify_all();
  }
}

void io_start(Store *s) {
  int n = 16;
  if (const char *e = getenv("DSV41_ENGRAM_IO_THREADS")) n = std::atoi(e);
  if (n < 1) n = 1;
  if (n > 64) n = 64;
  for (int i = 0; i < n; i++) s->io.threads.emplace_back(io_worker, s);
}

void io_stop(Store *s) {
  {
    std::lock_guard<std::mutex> lk(s->io.mu);
    s->io.stop = true;
  }
  s->io.cv.notify_all();
  for (auto &t : s->io.threads) t.join();
  s->io.threads.clear();
}

void io_run_batch(Store *s, UniqueMiss *batch, size_t count) {
  if (!count) return;
  // Prefill-sized miss batches run at high queue depth on sorted offsets (3-4x higher NVMe IOPS).
  if (count >= 256) {
    int nt = 64;
    if (const char *e = getenv("DSV41_ENGRAM_PREFILL_THREADS")) nt = std::atoi(e);
    nt = std::max(1, std::min(nt, 128));
    nt = std::min<int>(nt, int(count));
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    th.reserve(nt);
    for (int i = 0; i < nt; i++) {
      th.emplace_back([&, s] {
        pin_io_thread();
        for (size_t j; (j = next.fetch_add(1, std::memory_order_relaxed)) < count;) {
          batch[j].ok = read_row(s, uint64_t(batch[j].id), batch[j].row);
        }
      });
    }
    for (auto &t : th) t.join();
    return;
  }
  std::unique_lock<std::mutex> lk(s->io.mu);
  s->io.done_cv.wait(lk, [&] { return s->io.active == 0; });
  s->io.batch = batch;
  s->io.count = count;
  s->io.next.store(0, std::memory_order_relaxed);
  s->io.active = s->io.threads.size();
  s->io.gen++;
  lk.unlock();
  s->io.cv.notify_all();
  lk.lock();
  s->io.done_cv.wait(lk, [&] { return s->io.active == 0; });
}

struct Work {
  Store *store;
  const int64_t *ids;
  uint8_t *weights, *scales;
  uint64_t count;
};

bool read_bytes(Store *s, uint64_t offset, uint8_t *out, size_t length) {
  if (s->resident) {
    std::memcpy(out, s->resident + offset, length);
    return true;
  }
  alignas(kPage) static thread_local uint8_t page[16 * kPage];
  const uint64_t base = offset & ~uint64_t(kPage - 1);
  const size_t delta = offset - base;
  const size_t requested = ((delta + length + kPage - 1) / kPage) * kPage;
  if (requested > sizeof(page)) return false;
  ssize_t got;
  do {
    got = pread(s->fd, page, requested, base);
  } while (got < 0 && errno == EINTR);
  if (got < 0 || size_t(got) < delta + length) return false;
  std::memcpy(out, page + delta, length);
  s->reads.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool read_row(Store *s, uint64_t id, uint8_t *row) {
  if (!read_bytes(s, s->weight_offset + id * s->wbytes, row, s->wbytes)) return false;
  if (!s->sbytes) return true;
  if (s->scales_ram) {
    std::memcpy(row + s->wbytes, s->scales_ram + (id - s->row_lo) * s->sbytes, s->sbytes);
    return true;
  }
  return read_bytes(s, s->scale_offset + id * s->sbytes, row + s->wbytes, s->sbytes);
}

void scales_free(Store *s) {
  if (s->scales_ram) munmap(s->scales_ram, s->scales_ram_size);
  s->scales_ram = nullptr;
  s->scales_ram_size = 0;
}

void scales_load(Store *s, const char *path) {
  scales_free(s);
  const char *e = getenv("DSV41_ENGRAM_SCALES_RAM");
  if (s->resident || !s->sbytes || (e && std::strcmp(e, "0") == 0) || s->row_hi <= s->row_lo) return;
  const auto t0 = std::chrono::steady_clock::now();
  const size_t size = (s->row_hi - s->row_lo) * s->sbytes;
  void *m = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (m == MAP_FAILED) {
    std::fprintf(stderr, "row_store: scales in RAM: allocation failed errno=%d\n", errno);
    return;
  }
  madvise(m, size, MADV_HUGEPAGE);
  int fd = open(path, O_RDONLY | O_CLOEXEC);
  const uint64_t base = s->scale_offset + s->row_lo * s->sbytes;
  size_t done = 0;
  while (fd >= 0 && done < size) {
    const size_t n = std::min<size_t>(size - done, size_t(64) << 20);
    ssize_t got = pread(fd, static_cast<uint8_t *>(m) + done, n, base + done);
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) break;
    done += size_t(got);
  }
  if (fd >= 0) {
    posix_fadvise(fd, base, size, POSIX_FADV_DONTNEED);
    close(fd);
  }
  if (done < size) {
    std::fprintf(stderr, "row_store: scales in RAM: read %zu of %zu bytes, reading them per miss\n", done, size);
    munmap(m, size);
    return;
  }
  s->scales_ram = static_cast<uint8_t *>(m);
  s->scales_ram_size = size;
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::fprintf(stderr, "row_store: scales of rows [%llu,%llu) in RAM, %zu MiB in %.1f s\n",
               (unsigned long long)s->row_lo, (unsigned long long)s->row_hi, size >> 20, secs);
}

bool cache_has(Store *s, uint64_t id) {
  const uint64_t slot = id % s->slots;
  std::lock_guard<std::mutex> guard(s->locks[slot % kLockShards]);
  return s->keys[slot] == id + 1;
}

void cache_put(Store *s, uint64_t id, const uint8_t *row) {
  const uint64_t slot = id % s->slots;
  std::lock_guard<std::mutex> guard(s->locks[slot % kLockShards]);
  std::memcpy(s->cache + slot * s->row_bytes, row, s->row_bytes);
  s->keys[slot] = id + 1;
}

// Unified process-wide prewarm coordinator: runs queued layer jobs strictly in FIFO order
// (so Layer 1 finishes warming before Layer 14 starts), sorts & deduplicates offsets for
// ~3.5-4x higher NVMe IOPS, and yields whenever a foreground lookup is active.
struct WarmJob {
  Store *store;
  uint64_t done_at;
  std::vector<int64_t> ids;
};

struct GlobalWarm {
  std::thread thread;
  std::mutex mu;
  std::condition_variable cv, idle_cv;
  std::deque<WarmJob> queue;
  Store *current = nullptr;
  std::atomic<Store *> closing{nullptr};
  std::atomic<bool> stop_flag{false};
  bool started = false;
  bool stop = false;

  void shutdown() {
    std::thread to_join;
    {
      std::lock_guard<std::mutex> lk(mu);
      if (!started) return;
      stop = true;
      stop_flag.store(true, std::memory_order_release);
      queue.clear();
      to_join = std::move(thread);
      started = false;
    }
    cv.notify_all();
    if (to_join.joinable()) to_join.join();
    {
      std::lock_guard<std::mutex> lk(mu);
      stop = false;
      stop_flag.store(false, std::memory_order_release);
    }
  }

  ~GlobalWarm() { shutdown(); }
} g_warm;

void global_warm_loop() {
  pin_io_thread();
  int n = 64;
  if (const char *e = getenv("DSV41_ENGRAM_WARM_THREADS")) n = std::atoi(e);
  n = std::max(1, std::min(n, 128));
  for (;;) {
    WarmJob job;
    {
      std::unique_lock<std::mutex> lk(g_warm.mu);
      g_warm.current = nullptr;
      g_warm.idle_cv.notify_all();
      g_warm.cv.wait(lk, [&] { return g_warm.stop || !g_warm.queue.empty(); });
      if (g_warm.stop) return;
      job = std::move(g_warm.queue.front());
      g_warm.queue.pop_front();
      g_warm.current = job.store;
    }
    Store *s = job.store;
    for (int waited = 0; waited < 15000 && g_lookups.load(std::memory_order_acquire) < job.done_at; waited++) {
      if (g_warm.stop_flag.load(std::memory_order_acquire) ||
          g_warm.closing.load(std::memory_order_acquire) == s) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (g_warm.stop_flag.load(std::memory_order_acquire) ||
        g_warm.closing.load(std::memory_order_acquire) == s) continue;

    // Filter to owned range, sort ascending by row id (NVMe offset), and deduplicate.
    std::vector<uint64_t> uniq;
    uniq.reserve(job.ids.size());
    for (int64_t id : job.ids) {
      if (id >= int64_t(s->row_lo) && id < int64_t(s->row_hi)) uniq.push_back(uint64_t(id));
    }
    if (uniq.empty()) continue;
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<size_t> next{0};
    const int nt = std::min<int>(n, int(uniq.size()));
    auto work = [&] {
      pin_io_thread();
      uint8_t row[kMaxRow];
      for (size_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < uniq.size();) {
        if (g_warm.stop_flag.load(std::memory_order_acquire) ||
            g_warm.closing.load(std::memory_order_acquire) == s) return;
        // Yield NVMe bandwidth to any live foreground lookup.
        while (g_active_lookups.load(std::memory_order_relaxed) > 0) {
          if (g_warm.stop_flag.load(std::memory_order_acquire) ||
              g_warm.closing.load(std::memory_order_acquire) == s) return;
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const uint64_t id = uniq[i];
        if (cache_has(s, id)) continue;
        if (!read_row(s, id, row)) continue;
        cache_put(s, id, row);
        s->warm_rows.fetch_add(1, std::memory_order_relaxed);
      }
    };
    std::vector<std::thread> threads;
    threads.reserve(nt);
    for (int i = 0; i < nt; i++) threads.emplace_back(work);
    for (auto &t : threads) t.join();
    s->warm_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(),
        std::memory_order_relaxed);
  }
}

void warm_drain_store(Store *s) {
  std::unique_lock<std::mutex> lk(g_warm.mu);
  g_warm.closing.store(s, std::memory_order_release);
  auto it = g_warm.queue.begin();
  while (it != g_warm.queue.end()) {
    if (it->store == s) it = g_warm.queue.erase(it);
    else ++it;
  }
  g_warm.idle_cv.wait(lk, [&] { return g_warm.current != s; });
  g_warm.closing.store(nullptr, std::memory_order_release);
}

}  // namespace

extern "C" void row_store_warm(Store *s, const int64_t *ids, uint64_t count) {
  if (!s || !s->slots || s->resident || !count) return;
  std::vector<int64_t> batch(ids, ids + count);
  const uint64_t done_at = g_lookups.load(std::memory_order_acquire) + g_stores.load(std::memory_order_acquire);
  {
    std::lock_guard<std::mutex> lk(g_warm.mu);
    if (!g_warm.started) {
      g_warm.stop = false;
      g_warm.stop_flag.store(false, std::memory_order_release);
      g_warm.thread = std::thread(global_warm_loop);
      g_warm.started = true;
    }
    g_warm.queue.push_back(WarmJob{s, done_at, std::move(batch)});
  }
  g_warm.cv.notify_one();
}

extern "C" Store *row_store_open_fmt(const char *path, uint64_t rows, uint64_t woff, uint64_t soff,
                                     uint64_t budget, int resident_mode, uint64_t wbytes, uint64_t sbytes) {
  if (!wbytes || wbytes + sbytes > kMaxRow) {
    std::fprintf(stderr, "row_store_open: row of %llu + %llu bytes not supported\n",
                 (unsigned long long)wbytes, (unsigned long long)sbytes);
    return nullptr;
  }
  auto *s = new Store;
  s->wbytes = wbytes;
  s->sbytes = sbytes;
  s->row_bytes = wbytes + sbytes;
  s->fd = open(path, O_RDONLY | O_CLOEXEC | (resident_mode ? 0 : O_DIRECT));
  if (s->fd < 0) {
    std::fprintf(stderr, "row_store_open: open(%s) failed errno=%d\n", path, errno);
    delete s;
    return nullptr;
  }
  struct stat st;
  if (fstat(s->fd, &st) || woff > uint64_t(st.st_size) || soff > uint64_t(st.st_size) ||
      rows > (uint64_t(st.st_size) - woff) / wbytes || (sbytes && rows > (uint64_t(st.st_size) - soff) / sbytes)) {
    std::fprintf(stderr, "row_store_open: table extent invalid for %s\n", path);
    close(s->fd);
    delete s;
    return nullptr;
  }
  if (resident_mode) {
    s->resident_size = size_t(st.st_size);
    void *m = mmap(nullptr, s->resident_size, PROT_READ, MAP_SHARED | MAP_POPULATE, s->fd, 0);
    if (m == MAP_FAILED) {
      std::fprintf(stderr, "row_store_open: resident mmap failed errno=%d\n", errno);
      close(s->fd);
      delete s;
      return nullptr;
    }
    s->resident = static_cast<uint8_t *>(m);
    if (mlock(s->resident, s->resident_size)) {
      std::fprintf(stderr, "row_store_open: mlock failed errno=%d (need RLIMIT_MEMLOCK)\n", errno);
      munmap(s->resident, s->resident_size);
      close(s->fd);
      delete s;
      return nullptr;
    }
    budget = 0;
  }
  s->path = path;
  s->rows = rows;
  s->weight_offset = woff;
  s->scale_offset = soff;
  s->row_lo = 0;
  s->row_hi = rows;
  s->slots = budget / (s->row_bytes + sizeof(uint64_t));
  if (s->slots) {
    void *c = mmap(nullptr, s->slots * s->row_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    void *k = mmap(nullptr, s->slots * sizeof(uint64_t), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (c == MAP_FAILED || k == MAP_FAILED) {
      std::fprintf(stderr, "row_store_open: cache allocation failed errno=%d\n", errno);
      if (c != MAP_FAILED) munmap(c, s->slots * s->row_bytes);
      if (k != MAP_FAILED) munmap(k, s->slots * sizeof(uint64_t));
      close(s->fd);
      delete s;
      return nullptr;
    }
    s->cache = static_cast<uint8_t *>(c);
    s->keys = static_cast<uint64_t *>(k);
    madvise(s->cache, s->slots * s->row_bytes, MADV_HUGEPAGE);
  }
  if (!s->resident) io_start(s);
  g_stores.fetch_add(1);
  return s;
}

extern "C" Store *row_store_open(const char *path, uint64_t rows, uint64_t woff,
                                 uint64_t soff, uint64_t budget, int resident_mode) {
  return row_store_open_fmt(path, rows, woff, soff, budget, resident_mode, kDim, kScales);
}

extern "C" void row_store_lookup(void *opaque) {
  auto *work = static_cast<Work *>(opaque);
  Store *s = work->store;
  const auto t_call = std::chrono::steady_clock::now();
  const uint64_t wb = s->wbytes, sb = s->sbytes;

  struct RawMiss { int64_t id; uint64_t out_i; };
  std::vector<RawMiss> raw_misses;
  raw_misses.reserve(work->count > 64 ? work->count / 2 : work->count);

  for (uint64_t i = 0; i < work->count; ++i) {
    const int64_t id = work->ids[i];
    uint8_t *w_out = work->weights + i * wb;
    uint8_t *s_out = work->scales + i * sb;
    if (id < 0 || uint64_t(id) >= s->rows) {
      std::memset(w_out, 0, wb);
      if (sb) std::memset(s_out, 0, sb);
      s->errors.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    if (uint64_t(id) < s->row_lo || uint64_t(id) >= s->row_hi) {
      std::memset(w_out, 0, wb);
      if (sb) std::memset(s_out, 0, sb);
      continue;
    }
    const uint64_t slot = s->slots ? uint64_t(id) % s->slots : 0;
    bool hit = false;
    if (s->slots) {
      std::lock_guard<std::mutex> guard(s->locks[slot % kLockShards]);
      if (s->keys[slot] == uint64_t(id) + 1) {
        std::memcpy(w_out, s->cache + slot * s->row_bytes, wb);
        if (sb) std::memcpy(s_out, s->cache + slot * s->row_bytes + wb, sb);
        hit = true;
      }
    }
    if (hit) {
      s->hits.fetch_add(1, std::memory_order_relaxed);
    } else {
      raw_misses.push_back(RawMiss{id, i});
    }
  }

  if (!raw_misses.empty()) {
    const auto t_miss = std::chrono::steady_clock::now();
    // Sort misses by id (NVMe offset) and deduplicate identical ids before issuing I/O.
    if (raw_misses.size() > 1) {
      std::sort(raw_misses.begin(), raw_misses.end(),
                [](const RawMiss &a, const RawMiss &b) { return a.id < b.id; });
    }
    std::vector<UniqueMiss> uniq;
    uniq.reserve(raw_misses.size());
    std::vector<uint32_t> map_to_uniq(raw_misses.size());
    for (size_t k = 0; k < raw_misses.size(); ++k) {
      if (k == 0 || raw_misses[k].id != raw_misses[k - 1].id) {
        uniq.push_back(UniqueMiss{raw_misses[k].id, false, {}});
      }
      map_to_uniq[k] = uint32_t(uniq.size() - 1);
    }

    if (s->resident || uniq.size() == 1 || s->io.threads.empty()) {
      for (auto &u : uniq) u.ok = read_row(s, uint64_t(u.id), u.row);
    } else {
      g_active_lookups.fetch_add(1, std::memory_order_relaxed);
      io_run_batch(s, uniq.data(), uniq.size());
      g_active_lookups.fetch_sub(1, std::memory_order_relaxed);
    }
    s->miss_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_miss).count(),
        std::memory_order_relaxed);

    if (s->slots) {
      for (const auto &u : uniq) {
        if (!u.ok) continue;
        cache_put(s, uint64_t(u.id), u.row);
      }
    }
    for (size_t k = 0; k < raw_misses.size(); ++k) {
      const auto &u = uniq[map_to_uniq[k]];
      uint8_t *w_out = work->weights + raw_misses[k].out_i * wb;
      uint8_t *s_out = work->scales + raw_misses[k].out_i * sb;
      if (!u.ok) {
        std::memset(w_out, 0, wb);
        if (sb) std::memset(s_out, 0, sb);
        s->errors.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      s->misses.fetch_add(1, std::memory_order_relaxed);
      std::memcpy(w_out, u.row, wb);
      if (sb) std::memcpy(s_out, u.row + wb, sb);
    }
  }

  s->calls.fetch_add(1, std::memory_order_relaxed);
  g_lookups.fetch_add(1, std::memory_order_release);
  s->lookup_ns.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_call).count(),
      std::memory_order_relaxed);
}

extern "C" void row_store_stats(Store *s, uint64_t *out) {
  out[0] = s->hits.load();
  out[1] = s->misses.load();
  out[2] = s->reads.load();
  out[3] = s->slots * (s->row_bytes + sizeof(uint64_t));
  out[4] = s->errors.load();
  out[5] = s->slots;
  out[6] = s->calls.load();
  out[7] = s->lookup_ns.load();
  out[8] = s->miss_ns.load();
  out[9] = s->warm_rows.load();
  out[10] = s->warm_ns.load();
}

extern "C" void row_store_range(Store *s, uint64_t lo, uint64_t hi) {
  if (lo > hi || hi > s->rows) {
    std::fprintf(stderr, "row_store_range: invalid [%llu,%llu) for %llu rows\n",
                 (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)s->rows);
    return;
  }
  s->row_lo = lo;
  s->row_hi = hi;
  scales_load(s, s->path.c_str());
}

extern "C" void row_store_close(Store *s) {
  if (!s) return;
  const uint64_t rem = g_stores.fetch_sub(1) - 1;
  warm_drain_store(s);
  if (rem == 0) g_warm.shutdown();
  io_stop(s);
  scales_free(s);
  if (s->resident) munmap(s->resident, s->resident_size);
  if (s->slots) {
    munmap(s->cache, s->slots * s->row_bytes);
    munmap(s->keys, s->slots * sizeof(uint64_t));
  }
  if (s->fd >= 0) close(s->fd);
  delete s;
}

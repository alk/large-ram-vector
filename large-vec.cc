/* -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: nil -*- */
#include "large-vec.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#if __linux__
#include <linux/prctl.h>
#include <sys/prctl.h>
#endif  // __linux__

// MAP_NORESERVE is not defined on macOS. And is not part of the POSIX standard.
#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0
#endif  // !MAP_NORESERVE

namespace large_vec_internal {

void RaiseOOM() {
  fprintf(stderr, "RaiseOOM called!\n");
  abort();
}

namespace {

size_t RoundUp(size_t value, size_t alignment) {
  assert((alignment & (alignment - 1)) == 0);  // power of 2
  if ((value & (alignment - 1)) == 0) {
    return value;
  }
  if (value + alignment - 1 < value) {  // overflow
    large_vec_internal::RaiseOOM();
  }

  return (value + alignment - 1) & ~(alignment - 1);
}

uint8_t* MapChunk(size_t max_size) {
  max_size = RoundUp(max_size, getpagesize());
  void* mmap_result = mmap(0, max_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_NORESERVE | MAP_PRIVATE, -1, 0);
  if (mmap_result == MAP_FAILED) {
    large_vec_internal::RaiseOOM();
  }

#if defined(__linux__) && defined(PR_SET_VMA)
  (void)prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, reinterpret_cast<uintptr_t>(mmap_result), max_size, "large-vector");
#endif

  // printf("mmap-ed %g MiB at %p\n", double(max_size) / (1<<20), mmap_result);
  return static_cast<uint8_t*>(mmap_result);
}

void DecommitChunk(uint8_t* addr, size_t size) {
#ifndef NDEBUG
  size_t pagesize = getpagesize();
  assert((reinterpret_cast<uintptr_t>(addr) & (pagesize - 1)) == 0);
  assert((size & (pagesize - 1)) == 0);
#endif

#if __linux__
  // Linux is known to have a non-standard behavior of madvise(MADV_DONTNEED) with anonymous memory. It "removes" the
  // affected pages. On other OSes we use somewhat slower but standard "mmap-over" approach.
  if (madvise(addr, size, MADV_DONTNEED) != 0) {
    perror("madvise");
    abort();
  }
#else
  void* mmap_result = mmap(addr, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_FIXED, -1, 0);
  if (mmap_result != static_cast<void*>(addr)) {
    perror("mmap");
    abort();
  }
#endif
}

}  // anonymous namespace

MMChunk::MMChunk(MMChunkPool* pool, size_t max_size)
    : pool_(pool), base_addr_{MapChunk(max_size)}, max_size_{max_size}, in_use_size_{0}, realized_size_{0} {
  assert(pool_);
  pool_->live_chunks_count_.fetch_add(1, std::memory_order_relaxed);
}

MMChunk::~MMChunk() {
#ifndef NDEBUG
  {
    MutexLock lock(pool_->mutex_);
    assert(!trimmable_iter_.has_value());
  }
#endif

  pool_->live_chunks_count_.fetch_sub(1, std::memory_order_relaxed);
  SetInUseSize(0);
  SetRealizedSize(0);
  munmap(base_addr_, max_size_);
}

void MMChunk::ResizeTo(size_t new_size) {
  if (new_size > max_size_) {
    RaiseOOM();
  }

  MutexLock lock(gap_lock_);

  size_t pagesize = getpagesize();

  size_t want_realized = RoundUp(new_size, pagesize);
  if (want_realized > realized_size_) {
    pool_->ReclaimForChunk(this, want_realized - realized_size_);
    SetRealizedSize(want_realized);
  }
  SetInUseSize(new_size);

  size_t new_gap = realized_size_ - want_realized;
  pool_->UpdateTrimmable(this, new_gap > 0);
}

size_t MMChunk::Trim() {
#ifndef NDEBUG
  {
    MutexLock lock(pool_->mutex_);
    assert(!trimmable_iter_.has_value());
  }
#endif

  MutexLock lock(gap_lock_);
  size_t needed = RoundUp(in_use_size_, getpagesize());
  size_t freed = 0;

  if (needed < realized_size_) {
    freed = realized_size_ - needed;
    DecommitChunk(base_addr_ + needed, freed);
    SetRealizedSize(needed);
  }
  return freed;
}

MMChunkPool::Ptr MMChunkPool::Grab(size_t initial_size) {
  MMChunk* chunk = nullptr;
  {
    MutexLock lock(mutex_);
    if (!reclaimable_free_chunks_.empty()) {
      chunk = reclaimable_free_chunks_.back().release();
      reclaimable_free_chunks_.pop_back();
      assert(chunk->pool_ == this);
      chunk->pool_->mutex_.AssertHeld();
      chunk->is_used_ = true;
    } else if (!trimmed_free_chunks_.empty()) {
      chunk = trimmed_free_chunks_.front().release();
      trimmed_free_chunks_.pop_front();
      assert(chunk->pool_ == this);
      chunk->pool_->mutex_.AssertHeld();
      chunk->is_used_ = true;
    }
  }

  if (!chunk) {
    chunk = new MMChunk(this);
  }
  chunk->ResizeTo(initial_size);
  return {chunk, PoolDeleter{this}};
}

void MMChunkPool::Release(MMChunk* chunk) {
  {
    MutexLock lock(chunk->gap_lock_);
    chunk->SetInUseSize(0);
  }

  bool recycle = false;

  {
    MutexLock lock(mutex_);

    RemoveFromTrimmableLocked(chunk);

    if (reclaimable_free_chunks_.size() + trimmed_free_chunks_.size() < kMaxFreeChunks) {
      assert(chunk->pool_ == this);
      chunk->pool_->mutex_.AssertHeld();
      reclaimable_free_chunks_.emplace_back(chunk);
      recycle = true;
      chunk->is_used_ = false;
    }
  }

  if (!recycle) {
    delete chunk;
  }
}

void MMChunkPool::UpdateTrimmable(MMChunk* chunk, bool has_gap) {
  MutexLock lock(mutex_);
  assert(chunk->pool_ == this);
  chunk->pool_->mutex_.AssertHeld();  // note: extra indirections are needed for thread-safety analysis
  chunk->gap_lock_.AssertHeld();

  if (!has_gap) {
    RemoveFromTrimmableLocked(chunk);
    return;
  }

  assert(chunk->is_used_);
  assert(chunk->realized_size_ >= RoundUp(chunk->in_use_size_, getpagesize()));

  if (!chunk->trimmable_iter_.has_value()) {
    trimmable_used_chunks_.push_front(chunk);
    chunk->trimmable_iter_ = trimmable_used_chunks_.begin();
  } else {
    // LRU to the front
    trimmable_used_chunks_.splice(trimmable_used_chunks_.begin(), trimmable_used_chunks_, *chunk->trimmable_iter_);
  }
}

void MMChunkPool::RemoveFromTrimmableLocked(MMChunk* chunk) {
  assert(chunk->pool_ == this);
  chunk->pool_->mutex_.AssertHeld();
  if (chunk->trimmable_iter_.has_value()) {
    trimmable_used_chunks_.erase(*chunk->trimmable_iter_);
    chunk->trimmable_iter_ = std::nullopt;
  }
}

size_t MMChunkPool::Reclaim(size_t goal_bytes) { return ReclaimForChunk(nullptr, goal_bytes); }

size_t MMChunkPool::ReclaimForChunk(MMChunk* protected_chunk, size_t goal_bytes) {
  MutexLock lock(mutex_);

  if (protected_chunk) {
    RemoveFromTrimmableLocked(protected_chunk);
  }

  size_t reclaimed = 0;

  // 1. Trim reclaimable free chunks
  while (!reclaimable_free_chunks_.empty() && reclaimed < goal_bytes) {
    auto it = reclaimable_free_chunks_.begin();

    std::list<std::unique_ptr<MMChunk>> temp_list;
    temp_list.splice(temp_list.begin(), reclaimable_free_chunks_, it);

    mutex_.unlock();
    MMChunk* c = it->get();
    reclaimed += c->Trim();
    mutex_.lock();

    trimmed_free_chunks_.splice(trimmed_free_chunks_.begin(), temp_list);
  }

  // 2. Trim LRU trimmable active chunks
  while (!trimmable_used_chunks_.empty() && reclaimed < goal_bytes) {
    MMChunk* chunk = trimmable_used_chunks_.back();
    RemoveFromTrimmableLocked(chunk);

    mutex_.unlock();
    reclaimed += chunk->Trim();
    mutex_.lock();
  }

  return reclaimed;
}

}  // namespace large_vec_internal

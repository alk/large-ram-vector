// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: nil -*-
#ifndef LARGE_VEC_H_
#define LARGE_VEC_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>

#include "thread_annotations.h"

namespace large_vec_internal {

void RaiseOOM();

inline size_t MulOrOOM(size_t a, size_t b) {
  size_t res;
  if (__builtin_mul_overflow(a, b, &res)) {
    RaiseOOM();
  }
  return res;
}

class LOCKABLE Mutex : public std::mutex {
 public:
  void lock() EXCLUSIVE_LOCK_FUNCTION() { std::mutex::lock(); }
  void unlock() UNLOCK_FUNCTION() { std::mutex::unlock(); }
  void AssertHeld() ASSERT_EXCLUSIVE_LOCK() {}
};

class SCOPED_LOCKABLE MutexLock {
 public:
  explicit MutexLock(Mutex& mu) EXCLUSIVE_LOCK_FUNCTION(mu) : mu_(mu) { mu_.lock(); }
  ~MutexLock() UNLOCK_FUNCTION() { mu_.unlock(); }
  MutexLock(const MutexLock&) = delete;
  MutexLock& operator=(const MutexLock&) = delete;

 private:
  Mutex& mu_;
};

class MMChunk;

class MMChunkPool {
  friend class MMChunk;

 public:
  struct PoolDeleter {
    MMChunkPool* pool;
    PoolDeleter(MMChunkPool* pool) : pool(pool) {}
    void operator()(MMChunk* chunk) const;
  };

  using Ptr = std::unique_ptr<MMChunk, PoolDeleter>;
  static constexpr size_t kMaxFreeChunks = 1024;

  Ptr Grab(size_t initial_size);
  size_t Reclaim(size_t goal_bytes) LOCKS_EXCLUDED(mutex_);

  size_t total_in_use() const { return total_in_use_.load(std::memory_order_relaxed); }
  size_t total_realized() const { return total_realized_.load(std::memory_order_relaxed); }
  size_t live_chunks_count() const { return live_chunks_count_.load(std::memory_order_relaxed); }

 private:
  void Release(MMChunk* chunk);

  void RemoveFromTrimmableLocked(MMChunk* chunk) EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void UpdateTrimmable(MMChunk* chunk, bool has_gap) LOCKS_EXCLUDED(mutex_);
  size_t ReclaimForChunk(MMChunk* chunk, size_t goal_bytes) LOCKS_EXCLUDED(mutex_);

  mutable Mutex mutex_;

  std::atomic<size_t> total_in_use_ = 0;
  std::atomic<size_t> total_realized_ = 0;
  std::atomic<size_t> live_chunks_count_ = 0;
  std::list<MMChunk*> trimmable_used_chunks_ GUARDED_BY(mutex_);
  std::list<std::unique_ptr<MMChunk>> reclaimable_free_chunks_ GUARDED_BY(mutex_);
  std::list<std::unique_ptr<MMChunk>> trimmed_free_chunks_ GUARDED_BY(mutex_);
};

class MMChunk {
  friend class MMChunkPool;

 public:
  static constexpr size_t kDefaultMaxSize = 1 << 30;
  explicit MMChunk(MMChunkPool* pool, size_t max_size = kDefaultMaxSize);
  ~MMChunk();

  std::span<uint8_t> GetData() const { return std::span<uint8_t>{base_addr_, in_use_size_}; }

  void ResizeTo(size_t new_size_bytes) LOCKS_EXCLUDED(gap_lock_);

 private:
  // Takes the bytes between current_size_ and realized_size_ and
  // returns them to OS. Reduces realized_size_ accordingly.
  size_t Trim() LOCKS_EXCLUDED(gap_lock_);
  MMChunkPool* const pool_;
  // the starting address of the mapping
  uint8_t* const base_addr_;
  // total size of virtual address space reserved
  const size_t max_size_;

  Mutex gap_lock_ ACQUIRED_BEFORE(pool_->mutex_);  // This protects the gap between in_use_size_ and realized_size_

  // the number of bytes starting from base_addr_ currently in-use
  size_t in_use_size_;
  void SetInUseSize(size_t new_size) EXCLUSIVE_LOCKS_REQUIRED(gap_lock_);
  // the number of bytes starting from base_addr_ possibly touched by
  // this process.  Invariant: realized_size_ >= in_use_size_
  size_t realized_size_ GUARDED_BY(gap_lock_);
  void SetRealizedSize(size_t new_size) EXCLUSIVE_LOCKS_REQUIRED(gap_lock_);

  // "pointer" into pool's trimmable_used_chunks_ list
  std::optional<std::list<MMChunk*>::iterator> trimmable_iter_ GUARDED_BY(pool_->mutex_);
  // true if this chunk is in use by a vec
  bool is_used_ GUARDED_BY(pool_->mutex_) = true;
};

inline void MMChunkPool::PoolDeleter::operator()(MMChunk* chunk) const { pool->Release(chunk); }

inline void MMChunk::SetInUseSize(size_t new_size) {
  pool_->total_in_use_.fetch_add(new_size - in_use_size_, std::memory_order_relaxed);
  assert(pool_->total_in_use_.load() <= (~size_t{0} >> 1));  // do not overflow into "negative" values
  in_use_size_ = new_size;
}

inline void MMChunk::SetRealizedSize(size_t new_size) {
  pool_->total_realized_.fetch_add(new_size - realized_size_, std::memory_order_relaxed);
  assert(pool_->total_realized_.load() <= (~size_t{0} >> 1));  // do not overflow into "negative" values
  realized_size_ = new_size;
}

}  // namespace large_vec_internal

using MMChunkPool = large_vec_internal::MMChunkPool;

template <typename T>
class LargeVec {
 public:
  explicit LargeVec(MMChunkPool* pool) : LargeVec{pool, 0} {}

  template <class RandomAccessIt>
  LargeVec(MMChunkPool* pool, RandomAccessIt first, RandomAccessIt last) : LargeVec{pool, std::distance(first, last)} {
    std::uninitialized_copy(first, last, DataSpan().data());
    size_ = capacity_;
  }

  LargeVec(MMChunkPool* pool, size_t count, const T& value) : LargeVec{pool, count} {
    std::uninitialized_fill_n(DataSpan().data(), count, value);
    size_ = capacity_;
  }

  LargeVec(LargeVec&& other) noexcept
      : pool_(other.pool_),
        chunk_(std::move(other.chunk_)),
        data_(other.data_),
        size_(other.size_),
        capacity_(other.capacity_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
  }

  ~LargeVec() { std::destroy_n(DataSpan().data(), size_); }

  void swap(LargeVec& other) noexcept {
    using std::swap;
    swap(pool_, other.pool_);
    swap(chunk_, other.chunk_);
    swap(data_, other.data_);
    swap(size_, other.size_);
    swap(capacity_, other.capacity_);
  }

  friend void swap(LargeVec& a, LargeVec& b) noexcept { a.swap(b); }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  T& operator[](size_t index) { return DataSpan()[index]; }
  const T& operator[](size_t index) const { return DataSpan()[index]; }

  std::span<T> DataSpan() { return {data_, size_}; }

  std::span<const T> DataSpan() const { return {data_, size_}; }

  // Resizes the container to contain `new_size` elements.
  // If `new_size` is smaller than the current size, the content is reduced to its first `new_size` elements,
  // destroying those beyond.
  // NOTE: This does NOT necessarily free the underlying memory. Use ShrinkToFit() to release unused memory.
  void Resize(size_t new_size) {
    if (new_size < size_) {
      std::destroy(data_ + new_size, data_ + size_);
    } else if (new_size > size_) {
      Reserve(new_size);
      std::uninitialized_default_construct(data_ + size_, data_ + new_size);
    }
    size_ = new_size;
  }

  // Resizes the container to contain `new_size` elements.
  // If `new_size` is smaller than the current size, the content is reduced to its first `new_size` elements,
  // destroying those beyond.
  // NOTE: This does NOT necessarily free the underlying memory. Use ShrinkToFit() to release unused memory.
  void Resize(size_t new_size, const T& value) {
    if (new_size < size_) {
      std::destroy(data_ + new_size, data_ + size_);
    } else if (new_size > size_) {
      Reserve(new_size);
      std::uninitialized_fill(data_ + size_, data_ + new_size, value);
    }
    size_ = new_size;
  }
  void Reserve(size_t new_capacity) {
    if (new_capacity > capacity_) {
      if (!chunk_) {
        chunk_ = pool_->Grab(BytesFor(new_capacity));
        data_ = reinterpret_cast<T*>(chunk_->GetData().data());
      } else {
        chunk_->ResizeTo(BytesFor(new_capacity));
        // data_ remains valid (mmap invariant)
      }
      capacity_ = new_capacity;
    }
  }

  size_t capacity() const { return capacity_; }

  // Requests the removal of unused capacity.
  // It effectively frees the memory that is no longer needed.
  void ShrinkToFit() {
    if (capacity_ > size_) {
      if (size_ == 0) {
        chunk_.reset();  // Release the chunk if size is 0
        data_ = nullptr;
      } else if (chunk_) {
        chunk_->ResizeTo(BytesFor(size_));
      }
      capacity_ = size_;
    }
  }

  void PopBack() {
    std::destroy_at(DataSpan().data() + size_ - 1);
    size_--;
  }

  void Clear() {
    std::destroy_n(DataSpan().data(), size_);
    size_ = 0;
  }

  void PushBack(const T& value) {
    GrowIfNeeded();
    new (DataSpan().data() + size_) T(value);
    size_++;
  }

  void PushBack(T&& value) {
    GrowIfNeeded();
    new (DataSpan().data() + size_) T(std::move(value));
    size_++;
  }

  template <typename... Args>
  void EmplaceBack(Args&&... args) {
    GrowIfNeeded();
    new (DataSpan().data() + size_) T(std::forward<Args>(args)...);
    size_++;
  }

 private:
  void GrowIfNeeded() {
    if (size_ == capacity_) {
      size_t new_cap = capacity_ == 0 ? 4 : large_vec_internal::MulOrOOM(capacity_, 5) / 4;
      Reserve(new_cap);
    }
  }

  static size_t BytesFor(size_t elements) { return large_vec_internal::MulOrOOM(elements, sizeof(T)); }

  LargeVec(MMChunkPool* pool, size_t initial_capacity)
      : pool_(pool),
        chunk_{initial_capacity > 0 ? pool->Grab(BytesFor(initial_capacity))
                                    : MMChunkPool::Ptr(nullptr, MMChunkPool::PoolDeleter(pool))},
        data_{chunk_ ? reinterpret_cast<T*>(chunk_->GetData().data()) : nullptr},
        size_{0},
        capacity_{initial_capacity} {}

  MMChunkPool* pool_;
  MMChunkPool::Ptr chunk_;
  T* data_;
  size_t size_;
  size_t capacity_;
};

#endif  // LARGE_VEC_H_

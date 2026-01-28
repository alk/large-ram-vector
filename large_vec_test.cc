/* -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: nil -*- */
#include <gtest/gtest.h>

#include "large-vec.h"

TEST(LargeVecTest, ResizeLargerDefaultValue) {
  MMChunkPool pool;
  LargeVec<int> vec(&pool);
  ASSERT_EQ(vec.DataSpan().size(), 0);

  vec.Resize(100);
  ASSERT_EQ(vec.DataSpan().size(), 100);
  // Check elements are zero-initialized (default for int)
  for (int x : vec.DataSpan()) {
    ASSERT_EQ(x, 0);
  }

  // Resize even larger
  vec.Resize(200);
  ASSERT_EQ(vec.DataSpan().size(), 200);
  for (int x : vec.DataSpan()) {
    ASSERT_EQ(x, 0);
  }
}

TEST(LargeVecTest, ResizeLargerWithValue) {
  MMChunkPool pool;
  LargeVec<int> vec(&pool);
  vec.Resize(50, 42);
  ASSERT_EQ(vec.DataSpan().size(), 50);
  for (int x : vec.DataSpan()) {
    ASSERT_EQ(x, 42);
  }

  vec.Resize(100, 99);
  ASSERT_EQ(vec.DataSpan().size(), 100);
  for (size_t i = 0; i < 50; ++i) {
    ASSERT_EQ(vec.DataSpan()[i], 42);
  }
  for (size_t i = 50; i < 100; ++i) {
    ASSERT_EQ(vec.DataSpan()[i], 99);
  }
}

TEST(LargeVecTest, ResizeSmaller) {
  MMChunkPool pool;
  LargeVec<int> vec(&pool);
  vec.Resize(100, 10);
  ASSERT_EQ(vec.DataSpan().size(), 100);

  vec.Resize(50);
  ASSERT_EQ(vec.DataSpan().size(), 50);
  for (int x : vec.DataSpan()) {
    ASSERT_EQ(x, 10);
  }
}
TEST(LargeVecTest, PushBackAndGrowth) {
  MMChunkPool pool;
  LargeVec<int> vec(&pool);

  for (int i = 0; i < 100; ++i) {
    vec.PushBack(i);
  }

  ASSERT_EQ(vec.DataSpan().size(), 100);
  for (int i = 0; i < 100; ++i) {
    ASSERT_EQ(vec.DataSpan()[i], i);
  }
}

TEST(LargeVecTest, EmplaceBack) {
  struct Point {
    int x, y;
    Point(int x, int y) : x(x), y(y) {}
  };

  MMChunkPool pool;
  LargeVec<Point> vec(&pool);

  vec.EmplaceBack(1, 2);
  vec.EmplaceBack(3, 4);

  ASSERT_EQ(vec.size(), 2);
  ASSERT_FALSE(vec.empty());
  ASSERT_EQ(vec[0].x, 1);
  ASSERT_EQ(vec[0].y, 2);
  ASSERT_EQ(vec[1].x, 3);
  ASSERT_EQ(vec[1].y, 4);
}

TEST(LargeVecTest, Accessors) {
  MMChunkPool pool;
  LargeVec<int> vec(&pool);

  ASSERT_TRUE(vec.empty());
  ASSERT_EQ(vec.size(), 0);

  vec.PushBack(10);
  ASSERT_FALSE(vec.empty());
  ASSERT_EQ(vec.size(), 1);
  ASSERT_EQ(vec[0], 10);

  vec[0] = 20;
  ASSERT_EQ(vec[0], 20);

  const LargeVec<int>& const_vec = vec;
  ASSERT_EQ(const_vec.size(), 1);
  ASSERT_FALSE(const_vec.empty());
  ASSERT_EQ(const_vec[0], 20);
}

TEST(LargeVecTest, CapacityManagement) {
  MMChunkPool pool;
  LargeVec<int> vec(&pool);

  ASSERT_EQ(vec.capacity(), 0);
  vec.Reserve(100);
  ASSERT_GE(vec.capacity(), 100);

  vec.Resize(10);
  ASSERT_EQ(vec.size(), 10);
  ASSERT_GE(vec.capacity(), 100);

  vec.ShrinkToFit();
  ASSERT_EQ(vec.capacity(), 10);

  vec.Clear();
  ASSERT_EQ(vec.size(), 0);
  ASSERT_EQ(vec.capacity(), 10);
}

TEST(LargeVecTest, PopBack) {
  MMChunkPool pool;
  LargeVec<int> vec(&pool);

  vec.PushBack(1);
  vec.PushBack(2);
  ASSERT_EQ(vec.size(), 2);

  vec.PopBack();
  ASSERT_EQ(vec.size(), 1);
  ASSERT_EQ(vec[0], 1);

  vec.PopBack();
  ASSERT_TRUE(vec.empty());
}

TEST(LargeVecTest, TrimAndDecommit) {
  MMChunkPool pool;
  LargeVec<int> vec(&pool);

  // Grow to significant size (e.g. over a page)
  const int kCount = 10000;
  vec.Reserve(kCount);
  for (int i = 0; i < kCount; ++i) {
    vec.PushBack(i);
  }

  // Clear most
  vec.Resize(100);
  ASSERT_EQ(vec.size(), 100);
  // Capacity still high
  ASSERT_GE(vec.capacity(), kCount);

  // Shrink
  vec.ShrinkToFit();
  ASSERT_EQ(vec.capacity(), 100);

  // Verify remaining data is valid
  for (int i = 0; i < 100; ++i) {
    ASSERT_EQ(vec[i], i);
  }
}

TEST(LargeVecTest, PoolStats) {
  MMChunkPool pool;
  ASSERT_EQ(pool.total_in_use(), 0);
  ASSERT_EQ(pool.total_realized(), 0);

  {
    LargeVec<int> vec(&pool);
    ASSERT_EQ(pool.total_in_use(), 0);
    ASSERT_EQ(pool.total_realized(), 0);

    vec.PushBack(1);
    size_t expected_in_use = vec.capacity() * sizeof(int);
    ASSERT_EQ(pool.total_in_use(), expected_in_use);
    ASSERT_GT(pool.total_realized(), 0);  // At least one page

    size_t initial_realized = pool.total_realized();

    // Grow a lot
    vec.Resize(100000);  // 400KB
    ASSERT_GT(pool.total_in_use(), expected_in_use);
    size_t large_realized = pool.total_realized();
    ASSERT_GT(large_realized, initial_realized);

    // Shrink logical size
    vec.Resize(1);
    // In use does NOT drop for standard vector resize (capacity remains)
    ASSERT_EQ(pool.total_in_use(), 100000 * sizeof(int));
    // Realized should stay high
    ASSERT_EQ(pool.total_realized(), large_realized);

    // Trim
    vec.ShrinkToFit();
    // In use should drop now
    ASSERT_EQ(pool.total_in_use(), 1 * sizeof(int));
    // Realized stays high (lazy trim strategy)
    ASSERT_EQ(pool.total_realized(), large_realized);
  }

  // Destructor called
  // Chunk is pooled. In-use memory is cleared (logical size 0).
  ASSERT_EQ(pool.total_in_use(), 0);
  // Realized memory remains high (lazy trim).
  ASSERT_GT(pool.total_realized(), 0);
}

TEST(LargeVecTest, LiveChunkTracking) {
  MMChunkPool pool;
  ASSERT_EQ(pool.live_chunks_count(), 0);

  {
    LargeVec<int> vec1(&pool);
    // Lazy allocation: live count remains 0 until capacity needed.
    ASSERT_EQ(pool.live_chunks_count(), 0);
    vec1.Reserve(1);
    ASSERT_EQ(pool.live_chunks_count(), 1);

    LargeVec<int> vec2(&pool);
    vec2.PushBack(1);
    ASSERT_EQ(pool.live_chunks_count(), 2);

    {
      LargeVec<int> vec3(&pool);
      vec3.Reserve(1);
      ASSERT_EQ(pool.live_chunks_count(), 3);
    }
    // Chunk recycled into pool, still live.
    ASSERT_EQ(pool.live_chunks_count(), 3);
  }
  // All chunks recycled.
  ASSERT_EQ(pool.live_chunks_count(), 3);
}

TEST(LargeVecTest, Pooling) {
  MMChunkPool pool;
  ASSERT_EQ(pool.live_chunks_count(), 0);

  static constexpr size_t kMaxCount = large_vec_internal::MMChunkPool::kMaxFreeChunks;

  // Recycle test
  for (int i = 0; i < kMaxCount + 5; ++i) {
    {
      LargeVec<int> vec(&pool);
      vec.PushBack(i);
    }
    // After destruction, chunk goes to free pool.
    // It is still "live" (registered).
    ASSERT_EQ(pool.live_chunks_count(), 1);
  }

  // Capacity test
  {
    std::vector<LargeVec<int>> vecs;
    for (int i = 0; i < kMaxCount; ++i) {
      vecs.emplace_back(&pool);
      vecs.back().PushBack(i);
    }
    // 1 from before (if implemented correctly, maybe reused) + 14 new?
    // Wait, reusing:
    // We had 1 free chunk.
    // Iteration 0: Grab reuses free chunk. live=1.
    // Iteration 1: Grab creates new. live=2.
    // ...
    // Iteration 14: Grab creates new. live=15.
    ASSERT_EQ(pool.live_chunks_count(), kMaxCount);
  }

  // All destroyed.
  // Pool max size is kMaxCount.
  ASSERT_EQ(pool.live_chunks_count(), kMaxCount);

  // Create one more
  {
    LargeVec<int> vec(&pool);
    // Should reuse one from 10 free. live count constant.
    ASSERT_EQ(pool.live_chunks_count(), kMaxCount);
  }
  ASSERT_EQ(pool.live_chunks_count(), kMaxCount);
}

TEST(LargeVecTest, MoveAndSwap) {
  MMChunkPool pool;

  // Move Construction
  {
    LargeVec<int> vec1(&pool);
    vec1.PushBack(1);
    vec1.PushBack(2);

    // Move to vec2
    LargeVec<int> vec2(std::move(vec1));

    ASSERT_EQ(vec2.size(), 2);
    ASSERT_EQ(vec2[0], 1);
    ASSERT_EQ(vec2[1], 2);

    // vec1 should be reset
    ASSERT_EQ(vec1.size(), 0);
    ASSERT_EQ(vec1.capacity(), 0);
    ASSERT_TRUE(vec1.empty());

    // vec2 has the chunk. vec1 is lazy (null).
    // Live count should be 1.
    ASSERT_EQ(pool.live_chunks_count(), 1);
  }

  // Swap
  {
    LargeVec<int> vec1(&pool);
    vec1.PushBack(100);

    LargeVec<int> vec2(&pool);
    vec2.PushBack(200);
    vec2.PushBack(300);

    using std::swap;
    swap(vec1, vec2);

    ASSERT_EQ(vec1.size(), 2);
    ASSERT_EQ(vec1[0], 200);
    ASSERT_EQ(vec1[1], 300);

    ASSERT_EQ(vec2.size(), 1);
    ASSERT_EQ(vec2[0], 100);
  }
}

TEST(LargeVecTest, LRUReclamation) {
  MMChunkPool pool;
  size_t large_size = 100000;  // ~400KB

  // 1. vec1 grows large, then shrinks logically
  LargeVec<int> vec1(&pool);
  vec1.Resize(large_size);
  size_t peak_realized = pool.total_realized();
  ASSERT_GT(peak_realized, 0);

  vec1.Resize(1);
  vec1.ShrinkToFit();  // Sets capacity=1. Chunk still holds realized memory.
  ASSERT_EQ(pool.total_realized(), peak_realized);

  // 2. vec2 grows. Should trigger Reclaim.
  LargeVec<int> vec2(&pool);
  vec2.Resize(large_size);

  // vec2 growth should have triggered reclamation on vec1.
  // vec1 should be trimmed to page size.
  // vec2 should be at peak_realized.
  // Total should be peak + slightly more (for vec1's 1 element).

  size_t current_realized = pool.total_realized();
  ASSERT_LT(current_realized, peak_realized * 2);
  ASSERT_NEAR(current_realized, peak_realized + 4096, 4096);
}

TEST(LargeVecTest, DeadlockReproduction) {
  MMChunkPool pool;

  // 1. Create a vector and grow it to have some realized size
  LargeVec<int> vec(&pool);
  size_t large_size = 1000 * 1000;  // 4MB
  vec.Resize(large_size);

  // 2. Shrink it to create a gap and make it trimmable
  vec.Resize(100);
  vec.ShrinkToFit();
  // Now vec's chunk is in pool.trimmable_used_chunks_

  // 3. Resize to a larger size to trigger Reclaim
  // This triggers ReclaimForChunk(vec.chunk_), preventing self-trim deadlock
  size_t huge_size = large_size * 2;
  vec.Resize(huge_size);

  ASSERT_EQ(vec.size(), huge_size);
}

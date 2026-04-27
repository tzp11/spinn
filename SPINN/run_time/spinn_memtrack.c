/*
 * spinn_memtrack.c - 运行时内存追踪器实现
 *
 * 使用小型哈希表记录每次分配的 (ptr -> size) 映射，
 * 以便在 free 时准确扣减 current_alive。
 */

#include "spinn_memtrack.h"
#include <stdlib.h>
#include <string.h>

/* 哈希表: 记录 ptr -> size */
#define TRACK_BUCKETS 4096

typedef struct TrackEntry {
    void *ptr;
    size_t size;
    struct TrackEntry *next;
} TrackEntry;

static TrackEntry *g_table[TRACK_BUCKETS];
static SpinnMemStats g_stats;

static unsigned hash_ptr(void *p) {
    uintptr_t v = (uintptr_t)p;
    return (unsigned)((v >> 4) ^ (v >> 16)) & (TRACK_BUCKETS - 1);
}

static void record_alloc(void *ptr, size_t size) {
    if (!ptr) return;
    unsigned h = hash_ptr(ptr);
    TrackEntry *e = (TrackEntry*)malloc(sizeof(TrackEntry));
    e->ptr = ptr;
    e->size = size;
    e->next = g_table[h];
    g_table[h] = e;

    g_stats.alloc_count++;
    g_stats.total_allocated += size;
    g_stats.current_alive += size;
    if (g_stats.current_alive > g_stats.peak_alive)
        g_stats.peak_alive = g_stats.current_alive;
}

static size_t remove_alloc(void *ptr) {
    if (!ptr) return 0;
    unsigned h = hash_ptr(ptr);
    TrackEntry **pp = &g_table[h];
    while (*pp) {
        if ((*pp)->ptr == ptr) {
            TrackEntry *e = *pp;
            size_t sz = e->size;
            *pp = e->next;
            free(e);
            return sz;
        }
        pp = &(*pp)->next;
    }
    return 0; /* not tracked (e.g. tracker's own alloc) */
}

void spinn_memtrack_reset(void) {
    /* 清空哈希表 */
    for (int i = 0; i < TRACK_BUCKETS; i++) {
        TrackEntry *e = g_table[i];
        while (e) {
            TrackEntry *n = e->next;
            free(e);
            e = n;
        }
        g_table[i] = NULL;
    }
    memset(&g_stats, 0, sizeof(g_stats));
}

void spinn_memtrack_get(SpinnMemStats *out) {
    if (out) *out = g_stats;
}

void *spinn_tracked_malloc(size_t size) {
    void *p = malloc(size);
    record_alloc(p, size);
    return p;
}

void *spinn_tracked_calloc(size_t nmemb, size_t size) {
    void *p = calloc(nmemb, size);
    record_alloc(p, nmemb * size);
    return p;
}

void spinn_tracked_free(void *ptr) {
    size_t sz = remove_alloc(ptr);
    if (sz > 0) {
        g_stats.free_count++;
        g_stats.current_alive -= sz;
    }
    free(ptr);
}

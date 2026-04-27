/*
 * spinn_threadpool.c - 持久线程池（无锁版）
 *
 * 核心改进：
 *   1. 用 __atomic 原子操作代替 mutex 抢任务（零锁开销）
 *   2. worker 用 futex-style 自旋等待（低延迟唤醒）
 *   3. 主线程参与计算
 */

#include "spinn_threadpool.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdatomic.h>

/* worker 线程主循环 */
static void *worker_loop(void *arg) {
    SpinnThreadPool *pool = (SpinnThreadPool *)arg;
    
    for (;;) {
        /* 自旋等待新任务 (低延迟) */
        pthread_mutex_lock(&pool->mutex);
        int my_gen = pool->generation;
        while (!pool->shutdown && pool->generation == my_gen) {
            pthread_cond_wait(&pool->cond_work, &pool->mutex);
        }
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        pthread_mutex_unlock(&pool->mutex);
        
        /* 无锁抢任务 */
        for (;;) {
            int task = atomic_fetch_add(&pool->next_task, 1);
            if (task >= pool->total_tasks) break;
            
            pool->task_fn(pool->task_ctx, task, pool->total_tasks);
            atomic_fetch_add(&pool->tasks_done, 1);
        }
        
        /* 通知主线程（如果自己是最后完成的） */
        if (atomic_load(&pool->tasks_done) >= pool->total_tasks) {
            pthread_mutex_lock(&pool->mutex);
            pthread_cond_signal(&pool->cond_done);
            pthread_mutex_unlock(&pool->mutex);
        }
    }
    
    return NULL;
}

SpinnThreadPool *spinn_threadpool_create(int num_threads) {
    if (num_threads < 1) num_threads = 1;
    
    SpinnThreadPool *pool = calloc(1, sizeof(SpinnThreadPool));
    if (!pool) return NULL;
    
    int num_workers = num_threads - 1;
    pool->num_threads = num_threads;
    pool->shutdown = 0;
    pool->generation = 0;
    atomic_store(&pool->next_task, 0);
    atomic_store(&pool->tasks_done, 0);
    pool->total_tasks = 0;
    
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->cond_work, NULL);
    pthread_cond_init(&pool->cond_done, NULL);
    
    if (num_workers > 0) {
        pool->threads = malloc(num_workers * sizeof(pthread_t));
        for (int i = 0; i < num_workers; i++) {
            pthread_create(&pool->threads[i], NULL, worker_loop, pool);
        }
    }
    
    return pool;
}

void spinn_threadpool_parallel_for(SpinnThreadPool *pool,
                                    spinn_task_fn fn, void *ctx,
                                    int total_tasks) {
    if (!pool || total_tasks <= 0) return;
    
    /* 单任务或单线程: 直接执行 */
    if (total_tasks == 1 || pool->num_threads <= 1) {
        for (int i = 0; i < total_tasks; i++) fn(ctx, i, total_tasks);
        return;
    }
    
    /* 设置任务并唤醒 workers */
    pool->task_fn = fn;
    pool->task_ctx = ctx;
    pool->total_tasks = total_tasks;
    atomic_store(&pool->next_task, 0);
    atomic_store(&pool->tasks_done, 0);
    
    pthread_mutex_lock(&pool->mutex);
    pool->generation++;
    pthread_cond_broadcast(&pool->cond_work);
    pthread_mutex_unlock(&pool->mutex);
    
    /* 主线程也抢任务 */
    for (;;) {
        int task = atomic_fetch_add(&pool->next_task, 1);
        if (task >= total_tasks) break;
        fn(ctx, task, total_tasks);
        atomic_fetch_add(&pool->tasks_done, 1);
    }
    
    /* 等待所有任务完成 */
    pthread_mutex_lock(&pool->mutex);
    while (atomic_load(&pool->tasks_done) < total_tasks) {
        pthread_cond_wait(&pool->cond_done, &pool->mutex);
    }
    pthread_mutex_unlock(&pool->mutex);
}

void spinn_threadpool_destroy(SpinnThreadPool *pool) {
    if (!pool) return;
    
    int num_workers = pool->num_threads - 1;
    
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->cond_work);
    pthread_mutex_unlock(&pool->mutex);
    
    for (int i = 0; i < num_workers; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    free(pool->threads);
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->cond_work);
    pthread_cond_destroy(&pool->cond_done);
    free(pool);
}

/* 全局线程池 */
static SpinnThreadPool *g_pool = NULL;
static pthread_once_t g_pool_once = PTHREAD_ONCE_INIT;

static void init_global_pool(void) {
    const char *env = getenv("SPINN_NUM_THREADS");
    int n = env ? atoi(env) : 4;
    if (n < 1) n = 1;
    if (n > 32) n = 32;
    g_pool = spinn_threadpool_create(n);
}

static void destroy_global_pool(void) {
    if (g_pool) spinn_threadpool_destroy(g_pool);
}

SpinnThreadPool *spinn_get_global_pool(void) {
    pthread_once(&g_pool_once, init_global_pool);
    static int registered = 0;
    if (!registered) { atexit(destroy_global_pool); registered = 1; }
    return g_pool;
}

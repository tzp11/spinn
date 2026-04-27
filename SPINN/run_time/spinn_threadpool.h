/*
 * spinn_threadpool.h - 持久线程池（无锁版）
 */

#ifndef __SPINN_THREADPOOL_H__
#define __SPINN_THREADPOOL_H__

#include <pthread.h>
#include <stdatomic.h>

typedef void (*spinn_task_fn)(void *ctx, int task_id, int num_tasks);

typedef struct {
    pthread_t       *threads;
    int              num_threads;
    
    /* 任务 */
    spinn_task_fn    task_fn;
    void            *task_ctx;
    int              total_tasks;
    
    /* 无锁原子计数器 */
    atomic_int       next_task;
    atomic_int       tasks_done;
    
    /* 同步（仅用于唤醒/等待，非热路径） */
    pthread_mutex_t  mutex;
    pthread_cond_t   cond_work;
    pthread_cond_t   cond_done;
    
    int              shutdown;
    int              generation;
} SpinnThreadPool;

SpinnThreadPool *spinn_threadpool_create(int num_threads);
void spinn_threadpool_parallel_for(SpinnThreadPool *pool,
                                    spinn_task_fn fn, void *ctx,
                                    int total_tasks);
void spinn_threadpool_destroy(SpinnThreadPool *pool);
SpinnThreadPool *spinn_get_global_pool(void);

#endif

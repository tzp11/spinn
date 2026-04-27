/*
 * spinn_memtrack.h - 运行时内存追踪器
 * 
 * 用于统计推理过程中真实的内存分配行为:
 *   - 分配次数、释放次数
 *   - 当前存活字节、峰值存活字节
 *   - 总分配字节
 */

#ifndef __SPINN_MEMTRACK_H__
#define __SPINN_MEMTRACK_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t alloc_count;       /* malloc/calloc 调用次数 */
    uint32_t free_count;        /* free 调用次数 */
    size_t   total_allocated;   /* 累计分配字节 */
    size_t   current_alive;     /* 当前存活字节 */
    size_t   peak_alive;        /* 峰值存活字节 */
} SpinnMemStats;

void spinn_memtrack_reset(void);
void spinn_memtrack_get(SpinnMemStats *out);

void *spinn_tracked_malloc(size_t size);
void *spinn_tracked_calloc(size_t nmemb, size_t size);
void  spinn_tracked_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif /* __SPINN_MEMTRACK_H__ */

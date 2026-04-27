.intel_syntax noprefix
.global micro_6x16_asm_packed_a
.type micro_6x16_asm_packed_a, @function

// void micro_6x16_asm_packed_a(const float *pa, const float *pb, float *C, int ldc, int ck, int zero_mode)
// rdi = pa
// rsi = pb
// rdx = C
// rcx = ldc
// r8  = ck
// r9  = zero_mode
micro_6x16_asm_packed_a:
    shl rcx, 2      // ldc *= 4 (byte offset)

    test r9, r9
    jz .L_load_c

    // zero_mode == 1: Zero out YMM0-YMM11
    vpxor ymm0, ymm0, ymm0
    vpxor ymm1, ymm1, ymm1
    vpxor ymm2, ymm2, ymm2
    vpxor ymm3, ymm3, ymm3
    vpxor ymm4, ymm4, ymm4
    vpxor ymm5, ymm5, ymm5
    vpxor ymm6, ymm6, ymm6
    vpxor ymm7, ymm7, ymm7
    vpxor ymm8, ymm8, ymm8
    vpxor ymm9, ymm9, ymm9
    vpxor ymm10, ymm10, ymm10
    vpxor ymm11, ymm11, ymm11
    jmp .L_k_prep

.L_load_c:
    // zero_mode == 0: Load C[0..5, 0..15]
    vmovups ymm0, [rdx]
    vmovups ymm6, [rdx + 32]
    
    lea r10, [rdx + rcx]
    vmovups ymm1, [r10]
    vmovups ymm7, [r10 + 32]
    
    add r10, rcx
    vmovups ymm2, [r10]
    vmovups ymm8, [r10 + 32]
    
    add r10, rcx
    vmovups ymm3, [r10]
    vmovups ymm9, [r10 + 32]
    
    add r10, rcx
    vmovups ymm4, [r10]
    vmovups ymm10, [r10 + 32]
    
    add r10, rcx
    vmovups ymm5, [r10]
    vmovups ymm11, [r10 + 32]

.L_k_prep:
    mov r11, r8
    shr r11, 2          // k / 4
    jz .L_k_rem

    .align 16
.L_k_4_loop:
    // Step 0
    vmovups ymm12, [rsi + 0]
    vmovups ymm13, [rsi + 32]
    vbroadcastss ymm14, dword ptr [rdi + 0]
    vfmadd231ps ymm0, ymm12, ymm14
    vfmadd231ps ymm6, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 4]
    vfmadd231ps ymm1, ymm12, ymm14
    vfmadd231ps ymm7, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 8]
    vfmadd231ps ymm2, ymm12, ymm14
    vfmadd231ps ymm8, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 12]
    vfmadd231ps ymm3, ymm12, ymm14
    vfmadd231ps ymm9, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 16]
    vfmadd231ps ymm4, ymm12, ymm14
    vfmadd231ps ymm10, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 20]
    vfmadd231ps ymm5, ymm12, ymm14
    vfmadd231ps ymm11, ymm13, ymm14

    // Step 1
    vmovups ymm12, [rsi + 64]
    vmovups ymm13, [rsi + 96]
    vbroadcastss ymm14, dword ptr [rdi + 24]
    vfmadd231ps ymm0, ymm12, ymm14
    vfmadd231ps ymm6, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 28]
    vfmadd231ps ymm1, ymm12, ymm14
    vfmadd231ps ymm7, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 32]
    vfmadd231ps ymm2, ymm12, ymm14
    vfmadd231ps ymm8, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 36]
    vfmadd231ps ymm3, ymm12, ymm14
    vfmadd231ps ymm9, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 40]
    vfmadd231ps ymm4, ymm12, ymm14
    vfmadd231ps ymm10, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 44]
    vfmadd231ps ymm5, ymm12, ymm14
    vfmadd231ps ymm11, ymm13, ymm14

    // Step 2
    vmovups ymm12, [rsi + 128]
    vmovups ymm13, [rsi + 160]
    vbroadcastss ymm14, dword ptr [rdi + 48]
    vfmadd231ps ymm0, ymm12, ymm14
    vfmadd231ps ymm6, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 52]
    vfmadd231ps ymm1, ymm12, ymm14
    vfmadd231ps ymm7, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 56]
    vfmadd231ps ymm2, ymm12, ymm14
    vfmadd231ps ymm8, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 60]
    vfmadd231ps ymm3, ymm12, ymm14
    vfmadd231ps ymm9, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 64]
    vfmadd231ps ymm4, ymm12, ymm14
    vfmadd231ps ymm10, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 68]
    vfmadd231ps ymm5, ymm12, ymm14
    vfmadd231ps ymm11, ymm13, ymm14

    // Step 3
    vmovups ymm12, [rsi + 192]
    vmovups ymm13, [rsi + 224]
    vbroadcastss ymm14, dword ptr [rdi + 72]
    vfmadd231ps ymm0, ymm12, ymm14
    vfmadd231ps ymm6, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 76]
    vfmadd231ps ymm1, ymm12, ymm14
    vfmadd231ps ymm7, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 80]
    vfmadd231ps ymm2, ymm12, ymm14
    vfmadd231ps ymm8, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 84]
    vfmadd231ps ymm3, ymm12, ymm14
    vfmadd231ps ymm9, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 88]
    vfmadd231ps ymm4, ymm12, ymm14
    vfmadd231ps ymm10, ymm13, ymm14
    vbroadcastss ymm14, dword ptr [rdi + 92]
    vfmadd231ps ymm5, ymm12, ymm14
    vfmadd231ps ymm11, ymm13, ymm14

    add rdi, 96
    add rsi, 256
    dec r11
    jnz .L_k_4_loop

.L_k_rem:
    and r8, 3           // k % 4
    jz .L_store_c

.L_k_1_loop:
    vmovups ymm12, [rsi]
    vmovups ymm13, [rsi + 32]
    
    vbroadcastss ymm14, dword ptr [rdi + 0]
    vfmadd231ps ymm0, ymm12, ymm14
    vfmadd231ps ymm6, ymm13, ymm14
    
    vbroadcastss ymm14, dword ptr [rdi + 4]
    vfmadd231ps ymm1, ymm12, ymm14
    vfmadd231ps ymm7, ymm13, ymm14
    
    vbroadcastss ymm14, dword ptr [rdi + 8]
    vfmadd231ps ymm2, ymm12, ymm14
    vfmadd231ps ymm8, ymm13, ymm14
    
    vbroadcastss ymm14, dword ptr [rdi + 12]
    vfmadd231ps ymm3, ymm12, ymm14
    vfmadd231ps ymm9, ymm13, ymm14
    
    vbroadcastss ymm14, dword ptr [rdi + 16]
    vfmadd231ps ymm4, ymm12, ymm14
    vfmadd231ps ymm10, ymm13, ymm14
    
    vbroadcastss ymm14, dword ptr [rdi + 20]
    vfmadd231ps ymm5, ymm12, ymm14
    vfmadd231ps ymm11, ymm13, ymm14

    add rdi, 24
    add rsi, 64
    dec r8
    jnz .L_k_1_loop

.L_store_c:
    vmovups [rdx], ymm0
    vmovups [rdx + 32], ymm6
    
    lea r10, [rdx + rcx]
    vmovups [r10], ymm1
    vmovups [r10 + 32], ymm7
    
    add r10, rcx
    vmovups [r10], ymm2
    vmovups [r10 + 32], ymm8
    
    add r10, rcx
    vmovups [r10], ymm3
    vmovups [r10 + 32], ymm9
    
    add r10, rcx
    vmovups [r10], ymm4
    vmovups [r10 + 32], ymm10
    
    add r10, rcx
    vmovups [r10], ymm5
    vmovups [r10 + 32], ymm11

    vzeroupper
    ret

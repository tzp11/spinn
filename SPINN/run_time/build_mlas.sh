#!/bin/bash
# 编译 MLAS 库的核心部分

MLAS_SRC=/tmp/onnxruntime/onnxruntime/core/mlas/lib
MLAS_INC=/tmp/onnxruntime/onnxruntime/core/mlas/inc
ORT_ROOT=/tmp/onnxruntime
BUILD_DIR=mlas_build

# 创建输出目录和头文件结构
mkdir -p ${BUILD_DIR}/core/mlas/inc
cp ${MLAS_INC}/*.h ${BUILD_DIR}/core/mlas/inc/

# 编译选项
CXXFLAGS="-O3 -mavx2 -mfma -fopenmp -fPIC -std=c++17 -I${BUILD_DIR} -I${MLAS_SRC} -DONNX_NAMESPACE=onnx -DONNX_ML=1"

# 核心文件列表
CORE_FILES=(
    "platform.cpp"
    "sgemm.cpp"
)

echo "编译 MLAS 核心文件..."

# 编译每个文件
for file in "${CORE_FILES[@]}"; do
    if [ -f "${MLAS_SRC}/${file}" ]; then
        echo "  编译 ${file}..."
        g++ ${CXXFLAGS} -c "${MLAS_SRC}/${file}" -o "${BUILD_DIR}/${file%.cpp}.o" 2>&1 | head -5
    fi
done

# 创建静态库
echo "创建 libmlas.a..."
ar rcs ${BUILD_DIR}/libmlas.a ${BUILD_DIR}/*.o 2>/dev/null

if [ -f "${BUILD_DIR}/libmlas.a" ]; then
    echo "MLAS 库编译完成: ${BUILD_DIR}/libmlas.a"
    ls -lh ${BUILD_DIR}/libmlas.a
else
    echo "编译失败，检查错误信息"
fi

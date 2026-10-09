#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${QNN_SDK_ROOT:?Set QNN_SDK_ROOT to a licensed QAIRT SDK root (tested headers: 2.37)}"
api=${ANDROID_API:-28}
case "$api" in ''|*[!0-9]*) echo 'ANDROID_API must be numeric' >&2; exit 2;; esac
if ((api < 28)); then echo 'ANDROID_API must be >= 28' >&2; exit 2; fi
if [[ "${COLI_HEXAGON_NATIVE:-0}" == 1 ]]; then
    cc=$(command -v clang)
    ar=$(command -v llvm-ar)
    case "$("$cc" -dumpmachine)" in
        aarch64*-android*) ;;
        *) echo 'Native build requires an aarch64 Android clang (for example Termux).' >&2; exit 2;;
    esac
else
    : "${ANDROID_NDK_ROOT:?Set ANDROID_NDK_ROOT to an installed Android NDK}"
    host=${NDK_HOST_TAG:-linux-x86_64}
    toolchain="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/$host/bin"
    cc="$toolchain/aarch64-linux-android${api}-clang"
    ar="$toolchain/llvm-ar"
    test -x "$cc" && test -x "$ar"
fi
test -f "$QNN_SDK_ROOT/include/QNN/QnnInterface.h"
out=${COLI_HEXAGON_BUILD_DIR:-build-android}
mkdir -p "$out"
for source in coli_npu_buf coli_hexagon_engine coli_npu_qnn coli_npu_graph; do
    "$cc" -std=c99 -O2 -fPIC -Wall -Wextra -Werror -pthread \
        -I"$QNN_SDK_ROOT/include/QNN" -c "$source.c" -o "$out/$source.o"
done
"$ar" rcs "$out/libcoli_npu_hexagon.a" "$out/coli_npu_buf.o" \
    "$out/coli_hexagon_engine.o" "$out/coli_npu_qnn.o" "$out/coli_npu_graph.o"
"$cc" -std=c99 -O2 -Wall -Wextra -Werror -I"$QNN_SDK_ROOT/include/QNN" \
    probe_android.c "$out/libcoli_npu_hexagon.a" -ldl -pthread -o "$out/coli_hexagon_probe"
"$cc" -std=c99 -O2 -Wall -Wextra -Werror -DCOLI_NPU_TESTING -pthread \
    test_hexagon.c coli_npu_buf.c coli_hexagon_engine.c -ldl -lm -o "$out/test_hexagon"
printf '%s\n' "Built $out/libcoli_npu_hexagon.a (experimental support library)." \
    'No model graph, DSP kernel or Colibri inference integration is supplied by this build.'

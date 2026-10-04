// Export every supported ADES operator output immediately after execution.
//
// This is a diagnostic companion to ades_kernel_capture.cpp.  It deliberately
// uses the operator runtime's own descriptor copy and reloadable-vector writer,
// so it does not reinterpret tensor values or guess their layout.
//
// Build inside the digest-pinned CV22 ADES image:
//   g++ -std=c++17 -shared -fPIC -O2 -o /work/ades_operator_capture.so
//       /work/ades_operator_capture.cpp -ldl
// Run the native executor with:
//   env VERKEYE_OPERATOR_DUMP=/dump
//       LD_PRELOAD=/work/ades_operator_capture.so /work/ades-executor ...

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <link.h>
#include <string>

namespace {

// Recovered from the CV22 libades node_base::copy_op_output implementation.
// A pair is a 0x150-byte vp_descriptor followed by vector_data*.
constexpr std::size_t kOutputPairSize = 0x158;
constexpr std::size_t kVectorPointerOffset = 0x150;

struct RawVector {
  std::uint8_t* begin{};
  std::uint8_t* end{};
  std::uint8_t* capacity{};
};

static_assert(sizeof(RawVector) == 3 * sizeof(void*));

using Exec = void (*)(void*);
using CopyOutputs = void (*)(void*, RawVector*);
using WriteData = void (*)(const void*, std::FILE*);

std::atomic<unsigned> sequence{0};

template <typename Function>
Function resolve(const char* symbol) {
  auto function = reinterpret_cast<Function>(dlsym(RTLD_NEXT, symbol));
  if (function == nullptr) {
    std::fprintf(stderr, "VERKEYE_OPERATOR cannot resolve %s\n", symbol);
    std::abort();
  }
  return function;
}

struct LoadedLibrarySearch {
  const char* fragment;
  const char* path;
};

int find_loaded_library(dl_phdr_info* information, std::size_t, void* opaque) {
  auto* search = static_cast<LoadedLibrarySearch*>(opaque);
  if (information->dlpi_name != nullptr &&
      std::string(information->dlpi_name).find(search->fragment) !=
          std::string::npos) {
    search->path = information->dlpi_name;
    return 1;
  }
  return 0;
}

template <typename Function>
Function resolve_loaded(const char* library, const char* symbol) {
  LoadedLibrarySearch search{library, nullptr};
  dl_iterate_phdr(find_loaded_library, &search);
  if (search.path == nullptr) {
    std::fprintf(stderr, "VERKEYE_OPERATOR cannot locate loaded %s\n",
                 library);
    std::abort();
  }
  void* handle = dlopen(search.path, RTLD_NOW | RTLD_NOLOAD);
  if (handle == nullptr) {
    std::fprintf(stderr, "VERKEYE_OPERATOR cannot open provider %s\n",
                 search.path);
    std::abort();
  }
  auto function = reinterpret_cast<Function>(dlsym(handle, symbol));
  if (function == nullptr) {
    const char* error = dlerror();
    std::fprintf(stderr,
                 "VERKEYE_OPERATOR cannot resolve %s from %s: %s\n", symbol,
                 search.path, error == nullptr ? "unknown" : error);
    std::abort();
  }
  return function;
}

void dump_outputs(void* self, const char* kind) {
  const char* root = std::getenv("VERKEYE_OPERATOR_DUMP");
  if (root == nullptr || *root == '\0') {
    return;
  }
  static auto copy_outputs = resolve<CopyOutputs>(
      "_ZN9node_base14copy_op_outputERSt6vectorISt4pairI13vp_descriptorP11vector_dataESaIS5_EE");
  static auto write_data = resolve_loaded<WriteData>("libvamba_vec.so",
      "_ZNK11vector_data18write_data_to_fileEP8_IO_FILE");

  RawVector outputs;
  copy_outputs(self, &outputs);
  if (outputs.begin == nullptr || outputs.end < outputs.begin ||
      static_cast<std::size_t>(outputs.end - outputs.begin) %
              kOutputPairSize !=
          0) {
    std::fprintf(stderr, "VERKEYE_OPERATOR invalid output vector for %s\n",
                 kind);
    std::abort();
  }

  const auto call = sequence.fetch_add(1, std::memory_order_relaxed);
  const auto count =
      static_cast<std::size_t>(outputs.end - outputs.begin) /
      kOutputPairSize;
  for (std::size_t index = 0; index < count; ++index) {
    auto* entry = outputs.begin + index * kOutputPairSize;
    auto* vector = *reinterpret_cast<void**>(entry + kVectorPointerOffset);
    if (vector == nullptr) {
      std::fprintf(stderr,
                   "VERKEYE_OPERATOR null output call=%u kind=%s index=%zu\n",
                   call, kind, index);
      std::abort();
    }
    const std::string path = std::string(root) + "/operator-" +
                             std::to_string(call) + "-" + kind +
                             "-output-" + std::to_string(index) + ".bin";
    std::FILE* output = std::fopen(path.c_str(), "wb");
    if (output == nullptr) {
      std::fprintf(stderr, "VERKEYE_OPERATOR cannot create %s\n",
                   path.c_str());
      std::abort();
    }
    write_data(vector, output);
    if (std::fclose(output) != 0) {
      std::fprintf(stderr, "VERKEYE_OPERATOR cannot close %s\n",
                   path.c_str());
      std::abort();
    }
  }
  std::fprintf(stderr, "VERKEYE_OPERATOR call=%u kind=%s outputs=%zu\n", call,
               kind, count);
}

void execute_and_dump(void* self, const char* kind, const char* symbol) {
  auto original = resolve<Exec>(symbol);
  original(self);
  dump_outputs(self, kind);
}

}  // namespace

#define VERKEYE_CAPTURE_WRAPPER(wrapper, symbol_name, kind_name)           \
  extern "C" void wrapper(void* self) asm(symbol_name);                    \
  extern "C" void wrapper(void* self) {                                   \
    execute_and_dump(self, kind_name, symbol_name);                        \
  }

VERKEYE_CAPTURE_WRAPPER(verkeye_fastconv_exec,
                        "_ZN16node_fastconv_op4execEv", "fastconv")
VERKEYE_CAPTURE_WRAPPER(verkeye_madd_exec, "_ZN12node_madd_op4execEv",
                        "madd")
VERKEYE_CAPTURE_WRAPPER(verkeye_trans_exec, "_ZN13node_trans_op4execEv",
                        "trans")
VERKEYE_CAPTURE_WRAPPER(verkeye_shuffle_exec, "_ZN15node_shuffle_op4execEv",
                        "shuffle")
VERKEYE_CAPTURE_WRAPPER(verkeye_lvl_curve_exec,
                        "_ZN17node_lvl_curve_op4execEv", "lvlcurve")

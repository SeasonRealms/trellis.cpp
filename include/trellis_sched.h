#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace trellis {

struct Model;

// Multi-backend execution.
//
// trellis historically ran every graph with ggml_backend_graph_compute() on a single
// backend, which requires that backend to support *every* op in the graph. That holds for
// CUDA and Vulkan, whose op coverage is near-complete. It does not hold for the Qualcomm
// Hexagon NPU: the HTP skel implements a subset of ops and rejects tensors larger than
// VTCM, and with no scheduler those ops are simply never executed -- the run reports
// "dsp-rsp: UNKNOWN" and continues with undefined data until the DSP queue desynchronizes.
//
// ggml_backend_sched assigns each node to a backend that actually supports it, keeping
// weight-consuming ops on the backend that owns the weights and inserting copies where a
// node has to run elsewhere. That is what makes partial-coverage accelerators usable.
//
// Mode: -1 auto (on only for an HTP primary),
//        0 off (legacy single-backend path), 1 on.
void set_sched_mode(int mode);

// Add Vulkan between an HTP primary and the CPU catch-all. Mode -1 uses the
// TRELLIS_VULKAN_FALLBACK environment fallback, 0 disables it, and 1 enables it.
void set_vulkan_fallback_mode(int mode);

bool sched_enabled_for(ggml_backend_t primary);

// Fallback backends shared process-wide. Created on demand, never freed; they outlive the
// per-stage Models that borrow them.
ggml_backend_t fallback_cpu_backend();

// Build a scheduler for a freshly loaded model's backend, or nullptr when the single
// backend path is sufficient. Called by Model::load; the Model owns the result.
ggml_backend_sched_t make_sched(ggml_backend_t primary);

// Applies --threads / TRELLIS_THREADS to a CPU backend. Defined in trellis_model.cpp;
// declared here because the scheduler's CPU fallback needs it too -- a fallback stuck on
// ggml's 4-thread default would quietly become the bottleneck.
void tune_cpu_threads(ggml_backend_t b);

// --- backend registry -------------------------------------------------------
//
// ggml may be built with GGML_BACKEND_DL=ON, in which case ggml-cpu and ggml-cuda are
// separate modules that ggml.dll discovers at run time instead of libraries linked into
// this binary. ggml_backend_cpu_init() / ggml_backend_cuda_init() only exist in a DL=OFF
// link (and a DL=ON ggml does not even compile in GGML_USE_CUDA), so every entry point
// goes through the generic registry API instead: the three helpers below are the portable
// replacements, and ensure_backends_loaded() is the discovery step that makes the registry
// non-empty in the first place (a compiled no-op with DL=OFF).
//
// The registry reports whatever the ambient ggml.dll provides, which is how a trellis
// worker can share SeasonGGML's ggml-cpu-<variant>.dll + ggml-cuda.dll instead of
// shipping a second, differently configured copy of ggml.
void ensure_backends_loaded();

// Fresh CPU backend from the registry, or nullptr. Caller owns it.
ggml_backend_t cpu_backend_init();

// True when `b` runs on a CPU device. The answer comes from the registry, which is the
// only surface a DL=ON ggml.dll has: ggml_backend_is_cpu() belongs to ggml-cpu and is
// therefore neither declared for nor linkable against a dynamic-backend build.
bool backend_is_cpu(ggml_backend_t b);

// Fresh CUDA backend for CUDA device `gpu` (negative = device 0), or nullptr when the
// registry has no CUDA device. Caller owns it.
ggml_backend_t cuda_backend_init(int gpu);

// Allocation + execution strategy for one graph.
//
// Both paths follow the same contract, which mirrors what the call sites already did:
//   GraphExec ex(m);
//   ex.alloc(graph);                       // must precede input uploads
//   ggml_backend_tensor_set(input, ...);   // graph tensors exist only after alloc
//   ex.compute(graph, "label");
class GraphExec {
public:
    explicit GraphExec(const Model& m);
    ~GraphExec();

    GraphExec(const GraphExec&)            = delete;
    GraphExec& operator=(const GraphExec&) = delete;

    // Allocate a graph. Call again only when the graph itself changes.
    bool alloc(ggml_cgraph* graph);

    ggml_status compute(ggml_cgraph* graph, const char* label);

    // Bytes of compute buffer, for the existing [stage-alloc] diagnostics.
    size_t buffer_size() const;

    bool uses_sched() const { return sched_ != nullptr; }

private:
    ggml_backend_t       backend_ = nullptr;
    ggml_gallocr_t       galloc_  = nullptr;   // single-backend path
    ggml_backend_sched_t sched_   = nullptr;   // borrowed from the Model
};

}  // namespace trellis

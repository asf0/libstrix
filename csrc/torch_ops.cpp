// SPDX-License-Identifier: MIT
// The only file that registers ops with torch; kernels stay plain HIP + ATen.
#include <torch/all.h>
#include <torch/library.h>

namespace strix {
torch::Tensor gdn_gated_rmsnorm_h128_bf16(const torch::Tensor& x,
                                          const torch::Tensor& z,
                                          const torch::Tensor& weight,
                                          double eps, int64_t activation);
torch::Tensor _wmma_bf16_tile(const torch::Tensor& a, const torch::Tensor& b);
void moe_gemm_bf16(torch::Tensor& c, const torch::Tensor& a,
                   const torch::Tensor& b, const torch::Tensor& sorted_ids,
                   const torch::Tensor& expert_ids,
                   const torch::Tensor& num_post_padded,
                   const std::optional<torch::Tensor>& topk_weights,
                   int64_t top_k, int64_t block_m, int64_t block_n,
                   bool silu_mul, c10::string_view activation);
void moe_gemm_w4a16(torch::Tensor& c, const torch::Tensor& a,
                    const torch::Tensor& b, const torch::Tensor& scales,
                    const torch::Tensor& sorted_ids,
                    const torch::Tensor& expert_ids,
                    const torch::Tensor& num_post_padded,
                    const std::optional<torch::Tensor>& topk_weights,
                    int64_t top_k, int64_t block_m, int64_t block_n,
                    bool silu_mul, c10::string_view activation,
                    int64_t group_size);
void hc_up_gate_mix_bf16(torch::Tensor& y, const torch::Tensor& x,
                         const torch::Tensor& a, const torch::Tensor& w);
void qsa_union_attn_bf16(torch::Tensor& out, const torch::Tensor& q,
                         const torch::Tensor& key_cache,
                         const torch::Tensor& value_cache,
                         const torch::Tensor& indices,
                         const torch::Tensor& block_table);
void moe_act_quant_i8(const torch::Tensor& x, torch::Tensor& q, torch::Tensor& sa);
void moe_gemm_w4a8(torch::Tensor& c, const torch::Tensor& qa,
                   const torch::Tensor& sa, const torch::Tensor& w,
                   const torch::Tensor& scales, const torch::Tensor& sorted_ids,
                   const torch::Tensor& expert_ids,
                   const torch::Tensor& num_post_padded,
                   const std::optional<torch::Tensor>& topk_weights,
                   int64_t top_k, int64_t block_m, int64_t block_n,
                   bool silu_mul, c10::string_view activation);
void moe_gemm_w8a16(torch::Tensor& c, const torch::Tensor& a,
                    const torch::Tensor& b, const torch::Tensor& scales,
                    const torch::Tensor& sorted_ids,
                    const torch::Tensor& expert_ids,
                    const torch::Tensor& num_post_padded,
                    const std::optional<torch::Tensor>& topk_weights,
                    int64_t top_k, int64_t block_m, int64_t block_n,
                    bool silu_mul, c10::string_view activation);
void gdn_chunk_fwd_k128_v128_c64_bf16(
    torch::Tensor& o, torch::Tensor& final_state, torch::Tensor& a,
    torch::Tensor& p, torch::Tensor& gc, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& g,
    const torch::Tensor& beta, const torch::Tensor& initial_state,
    const torch::Tensor& cu_seqlens, const torch::Tensor& chunk_offsets,
    int64_t max_chunks, double scale);
void attn_prefill32_bf16(torch::Tensor& out, const torch::Tensor& q,
                         const torch::Tensor& key_cache,
                         const torch::Tensor& value_cache,
                         const torch::Tensor& cu_seqlens_q,
                         const torch::Tensor& seq_lens,
                         const torch::Tensor& block_table,
                         const torch::Tensor& tile_offsets, int64_t max_tiles,
                         double scale);
void attn_prefill_bf16(torch::Tensor& out, const torch::Tensor& q,
                            const torch::Tensor& key_cache,
                            const torch::Tensor& value_cache,
                            const std::optional<torch::Tensor>& k_scale,
                            const std::optional<torch::Tensor>& v_scale,
                            const torch::Tensor& cu_seqlens_q,
                            const torch::Tensor& seq_lens,
                            const torch::Tensor& block_table,
                            const torch::Tensor& tile_offsets,
                            int64_t max_tiles, double scale,
                            int64_t num_splits, int64_t window_left);
void attn_decode_bf16(torch::Tensor& out, const torch::Tensor& q,
                      const torch::Tensor& key_cache,
                      const torch::Tensor& value_cache,
                      const torch::Tensor& seq_lens,
                      const torch::Tensor& block_table, double scale,
                      int64_t num_splits, int64_t window_left,
                      const std::optional<torch::Tensor>& k_scale,
                      const std::optional<torch::Tensor>& v_scale);
torch::Tensor w4a16_wmma_gemm(const torch::Tensor& x, const torch::Tensor& w,
                              const torch::Tensor& scales,
                              const std::optional<torch::Tensor>& bias,
                              int64_t group_size);
void gdn_decode_mtp_bf16(
    const torch::Tensor& mixed_qkv, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_log, const torch::Tensor& dt_bias,
    const torch::Tensor& state_indices, const torch::Tensor& cu_seqlens,
    const torch::Tensor& num_accepted_tokens, torch::Tensor& state,
    const torch::Tensor& output_gate, const torch::Tensor& norm_weight,
    torch::Tensor& out, double scale, double norm_eps,
    c10::string_view output_gate_activation);
torch::Tensor w8a16_skinny_gemm(const at::Tensor& in_a, const at::Tensor& in_b,
                                const at::Tensor& in_scale,
                                const std::optional<at::Tensor>& in_bias,
                                const int64_t CuCount, const int64_t group_size);
void gdn_causal_conv1d_w4_bf16(torch::Tensor& out, const torch::Tensor& x,
                               const torch::Tensor& weight,
                               const std::optional<torch::Tensor>& bias,
                               torch::Tensor& conv_states,
                               const torch::Tensor& cu_seqlens,
                               const torch::Tensor& cache_indices,
                               const std::optional<torch::Tensor>& has_initial_state,
                               bool silu);
void gdn_post_conv_bf16(torch::Tensor& q, torch::Tensor& k, torch::Tensor& v,
                        torch::Tensor& g, torch::Tensor& beta,
                        const torch::Tensor& conv, const torch::Tensor& a,
                        const torch::Tensor& b, const torch::Tensor& a_log,
                        const torch::Tensor& dt_bias, double eps);
torch::Tensor w8a8_wmma_gemm(const at::Tensor& in_w8, const at::Tensor& in_act,
                             const at::Tensor& in_scale);
torch::Tensor w8a8_wmma_gemm_cfg(const at::Tensor& in_w8,
                                 const at::Tensor& in_act,
                                 const at::Tensor& in_scale, int64_t cfg);
torch::Tensor w4a8_wmma_gemm(const at::Tensor& in_w4, const at::Tensor& in_act,
                             const at::Tensor& in_scale);
torch::Tensor mxfp4_wmma_gemm(const torch::Tensor& w, const torch::Tensor& x,
                              const torch::Tensor& scales,
                              const std::optional<torch::Tensor>& bias);
torch::Tensor mxfp4_wmma_gemm_cfg(const torch::Tensor& w, const torch::Tensor& x,
                                  const torch::Tensor& scales,
                                  const std::optional<torch::Tensor>& bias,
                                  int64_t cfg);
torch::Tensor mxfp4_dequant(const torch::Tensor& w, const torch::Tensor& scales);
torch::Tensor w4a16_dequant(const torch::Tensor& w, const torch::Tensor& scales,
                            int64_t group_size);
torch::Tensor mxfp4_gemv(const torch::Tensor& x, const torch::Tensor& w,
                         const torch::Tensor& scales);
}

TORCH_LIBRARY(strix, m) {
  m.def(
      "gdn_gated_rmsnorm_h128_bf16(Tensor x, Tensor z, Tensor weight, "
      "float eps, int activation) -> Tensor");
  m.def("_wmma_bf16_tile(Tensor a, Tensor b) -> Tensor");
  // `activation` (only read when silu_mul) selects the fused epilogue's gate.
  m.def(
      "moe_gemm_bf16(Tensor(a!) c, Tensor a, Tensor b, Tensor sorted_ids, "
      "Tensor expert_ids, Tensor num_post_padded, Tensor? topk_weights, "
      "int top_k, int block_m, int block_n, bool silu_mul=False, "
      "str activation='silu') -> ()");
  m.def(
      "moe_gemm_w4a16(Tensor(a!) c, Tensor a, Tensor b, Tensor scales, "
      "Tensor sorted_ids, Tensor expert_ids, Tensor num_post_padded, "
      "Tensor? topk_weights, int top_k, int block_m, int block_n, "
      "bool silu_mul=False, str activation='silu', int group_size=32) -> ()");
  // Qwen3.8-Flash-Next hyper-connection up GEMM fused with the gated mix.
  m.def("hc_up_gate_mix_bf16(Tensor(a!) y, Tensor x, Tensor a, Tensor w) -> ()");
  // Qwen3.8-Flash-Next QSA sparse attention over row unions (single request).
  m.def(
      "qsa_union_attn_bf16(Tensor(a!) out, Tensor q, Tensor key_cache, "
      "Tensor value_cache, Tensor indices, Tensor block_table) -> ()");
  // int8 activations per (row, 128-element group) for moe_gemm_w4a8.
  m.def("moe_act_quant_i8(Tensor x, Tensor(a!) q, Tensor(b!) sa) -> ()");
  m.def(
      "moe_gemm_w4a8(Tensor(a!) c, Tensor qa, Tensor sa, Tensor b, Tensor scales, "
      "Tensor sorted_ids, Tensor expert_ids, Tensor num_post_padded, "
      "Tensor? topk_weights, int top_k, int block_m, int block_n, "
      "bool silu_mul=False, str activation='silu') -> ()");
  m.def(
      "moe_gemm_w8a16(Tensor(a!) c, Tensor a, Tensor b, Tensor scales, "
      "Tensor sorted_ids, Tensor expert_ids, Tensor num_post_padded, "
      "Tensor? topk_weights, int top_k, int block_m, int block_n, "
      "bool silu_mul=False, str activation='silu') -> ()");
  m.def(
      "gdn_chunk_fwd_k128_v128_c64_bf16(Tensor(a!) o, Tensor(b!) final_state, "
      "Tensor(c!) a, Tensor(d!) p, Tensor(e!) gc, "
      "Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
      "Tensor initial_state, Tensor cu_seqlens, Tensor chunk_offsets, "
      "int max_chunks, float scale) -> ()");
  m.def(
      "attn_prefill_bf16(Tensor(a!) out, Tensor q, Tensor key_cache, "
      "Tensor value_cache, Tensor? k_scale, Tensor? v_scale, "
      "Tensor cu_seqlens_q, Tensor seq_lens, "
      "Tensor block_table, Tensor tile_offsets, int max_tiles, float scale, "
      "int num_splits, int window_left=-1) -> ()");
  m.def(
      "attn_prefill32_bf16(Tensor(a!) out, Tensor q, Tensor key_cache, "
      "Tensor value_cache, Tensor cu_seqlens_q, Tensor seq_lens, "
      "Tensor block_table, Tensor tile_offsets, int max_tiles, float scale)"
      " -> ()");
  m.def(
      "w8a16_skinny_gemm(Tensor in_a, Tensor in_b, Tensor in_scale, "
      "Tensor? in_bias, int CuCount, int group_size) -> Tensor");
  m.def(
      "attn_decode_bf16(Tensor(a!) out, Tensor q, Tensor key_cache, "
      "Tensor value_cache, Tensor seq_lens, Tensor block_table, float scale, "
      "int num_splits, int window_left=-1, Tensor? k_scale=None, "
      "Tensor? v_scale=None) -> ()");
  m.def(
      "w4a16_wmma_gemm(Tensor x, Tensor w, Tensor scales, Tensor? bias, "
      "int group_size) -> Tensor");
  // Same schema as vLLM's _C::fused_gdn_decode_post_conv_mtp.
  m.def(
      "gdn_decode_mtp_bf16("
      "Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
      "Tensor state_indices, Tensor cu_seqlens, Tensor num_accepted_tokens, "
      "Tensor! state, Tensor output_gate, Tensor norm_weight, Tensor! out, "
      "float scale, float norm_eps=1e-5, "
      "str output_gate_activation='silu') -> ()");
  m.def("w8a8_wmma_gemm(Tensor in_w8, Tensor in_act, Tensor in_scale) -> Tensor");
  // Bench-only: the same GEMM on one of the fixed tile configs, for
  // bench/kernels/w8a8_gemm.py. M must be a multiple of 128.
  m.def(
      "w8a8_wmma_gemm_cfg(Tensor in_w8, Tensor in_act, Tensor in_scale, "
      "int cfg) -> Tensor");
  m.def("w4a8_wmma_gemm(Tensor in_w4, Tensor in_act, Tensor in_scale) -> Tensor");
  // Native MXFP4 W4A16: packed e2m1 weights read in place, bf16 activations.
  m.def(
      "mxfp4_wmma_gemm(Tensor w, Tensor x, Tensor scales, Tensor? bias) "
      "-> Tensor");
  // Bench-only: the same GEMM on one fixed tile config, for
  // bench/kernels/mxfp4_gemm.py.
  m.def(
      "mxfp4_wmma_gemm_cfg(Tensor w, Tensor x, Tensor scales, Tensor? bias, "
      "int cfg) -> Tensor");
  m.def("mxfp4_gemv(Tensor x, Tensor w, Tensor scales) -> Tensor");
  m.def("mxfp4_dequant(Tensor w, Tensor scales) -> Tensor");
  m.def("w4a16_dequant(Tensor w, Tensor scales, int group_size) -> Tensor");
  m.def(
      "gdn_causal_conv1d_w4_bf16(Tensor(a!) out, Tensor x, Tensor weight, "
      "Tensor? bias, Tensor(b!) conv_states, Tensor cu_seqlens, "
      "Tensor cache_indices, Tensor? has_initial_state, bool silu) -> ()");
  m.def(
      "gdn_post_conv_bf16(Tensor(a!) q, Tensor(b!) k, Tensor(c!) v, "
      "Tensor(d!) g, Tensor(e!) beta, Tensor conv, Tensor a, Tensor b, "
      "Tensor a_log, Tensor dt_bias, float eps) -> ()");
}

TORCH_LIBRARY_IMPL(strix, CUDA, m) {
  m.impl("gdn_gated_rmsnorm_h128_bf16", &strix::gdn_gated_rmsnorm_h128_bf16);
  m.impl("_wmma_bf16_tile", &strix::_wmma_bf16_tile);
  m.impl("moe_gemm_bf16", &strix::moe_gemm_bf16);
  m.impl("moe_gemm_w4a16", &strix::moe_gemm_w4a16);
  m.impl("moe_act_quant_i8", &strix::moe_act_quant_i8);
  m.impl("qsa_union_attn_bf16", &strix::qsa_union_attn_bf16);
  m.impl("hc_up_gate_mix_bf16", &strix::hc_up_gate_mix_bf16);
  m.impl("moe_gemm_w4a8", &strix::moe_gemm_w4a8);
  m.impl("moe_gemm_w8a16", &strix::moe_gemm_w8a16);
  m.impl("gdn_chunk_fwd_k128_v128_c64_bf16",
         &strix::gdn_chunk_fwd_k128_v128_c64_bf16);
  m.impl("attn_prefill_bf16", &strix::attn_prefill_bf16);
  m.impl("attn_prefill32_bf16", &strix::attn_prefill32_bf16);
  m.impl("w8a16_skinny_gemm", &strix::w8a16_skinny_gemm);
  m.impl("w8a8_wmma_gemm", &strix::w8a8_wmma_gemm);
  m.impl("w8a8_wmma_gemm_cfg", &strix::w8a8_wmma_gemm_cfg);
  m.impl("gdn_decode_mtp_bf16", &strix::gdn_decode_mtp_bf16);
  m.impl("w4a16_wmma_gemm", &strix::w4a16_wmma_gemm);
  m.impl("attn_decode_bf16", &strix::attn_decode_bf16);
  m.impl("w4a8_wmma_gemm", &strix::w4a8_wmma_gemm);
  m.impl("mxfp4_wmma_gemm", &strix::mxfp4_wmma_gemm);
  m.impl("mxfp4_wmma_gemm_cfg", &strix::mxfp4_wmma_gemm_cfg);
  m.impl("mxfp4_gemv", &strix::mxfp4_gemv);
  m.impl("mxfp4_dequant", &strix::mxfp4_dequant);
  m.impl("w4a16_dequant", &strix::w4a16_dequant);
  m.impl("gdn_causal_conv1d_w4_bf16", &strix::gdn_causal_conv1d_w4_bf16);
  m.impl("gdn_post_conv_bf16", &strix::gdn_post_conv_bf16);
}

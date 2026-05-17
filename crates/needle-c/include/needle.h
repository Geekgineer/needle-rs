/**
 * needle.h — C ABI for the Needle SAN inference engine.
 *
 * Usage (compile against the shared library):
 *   gcc -o my_program my_program.c -lneedle -L./target/release
 *
 * Python ctypes example:
 *   import ctypes, os
 *   lib = ctypes.CDLL("./target/release/libneedle_c.so")
 *   lib.needle_load.restype = ctypes.c_void_p
 *   lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
 *   handle = lib.needle_load(b"weights/needle.safetensors", b"weights/vocab.txt")
 *   lib.needle_run.restype = ctypes.c_char_p
 *   lib.needle_run.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
 *   result = lib.needle_run(handle, b"What's the weather?", b'[{"name":"get_weather",...}]')
 *   print(result.decode())
 *   lib.needle_free(handle)
 */

#ifndef NEEDLE_H
#define NEEDLE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle returned by needle_load / needle_load_bytes. */
typedef struct NeedleHandle NeedleHandle;

/**
 * Streaming callback type for needle_run_stream.
 *
 * @param token_id  Raw token ID of the generated token.
 * @param piece     Null-terminated UTF-8 text for this token (e.g. " Paris").
 * @param user_data Caller-supplied context pointer (passed through unchanged).
 */
typedef void (*NeedleStreamCallback)(uint32_t token_id,
                                     const char *piece,
                                     void *user_data);

/* ── Loading ──────────────────────────────────────────────────────────────── */

/**
 * Load a Needle model from weight and vocabulary files on disk.
 *
 * @param weights_path  Path to .safetensors weight file.
 * @param vocab_path    Path to vocabulary text file (one piece per line).
 * @return Opaque handle on success, NULL on failure.
 *         Call needle_last_error() to retrieve the error message.
 *         Caller must free the handle with needle_free().
 */
NeedleHandle *needle_load(const char *weights_path, const char *vocab_path);

/**
 * Load a Needle model from in-memory byte buffers.
 *
 * Useful for loading weights fetched from a network, embedded as binary
 * resources, or passed from Python/Go without writing to disk.
 *
 * @param weights_data  Pointer to raw SafeTensors file bytes.
 * @param weights_len   Length of weights_data in bytes.
 * @param vocab_data    Pointer to UTF-8 vocabulary text bytes.
 * @param vocab_len     Length of vocab_data in bytes.
 * @return Opaque handle on success, NULL on failure.
 *         Caller must free the handle with needle_free().
 */
NeedleHandle *needle_load_bytes(const uint8_t *weights_data, size_t weights_len,
                                const uint8_t *vocab_data,   size_t vocab_len);

/* ── Inference ────────────────────────────────────────────────────────────── */

/**
 * Run single-example inference.
 *
 * @param handle     Engine handle from needle_load / needle_load_bytes.
 * @param query      Null-terminated UTF-8 query string.
 * @param tools_json Null-terminated UTF-8 JSON array of tool definitions.
 * @return Heap-allocated null-terminated UTF-8 output string.
 *         Returns NULL on error. Caller must free with needle_free_str().
 *
 * Output format: JSON string such as
 *   [{"name":"get_weather","arguments":{"location":"Paris"}}]
 */
char *needle_run(NeedleHandle *handle, const char *query, const char *tools_json);

/**
 * Run inference with per-token streaming callback.
 *
 * Identical to needle_run but fires `callback` for each generated token
 * before returning the final post-processed output string.
 *
 * @param callback  Function called for each token (may be NULL to skip streaming).
 * @param user_data Passed through unchanged to each callback invocation.
 * @return Same semantics as needle_run. Caller must free with needle_free_str().
 */
char *needle_run_stream(NeedleHandle    *handle,
                        const char      *query,
                        const char      *tools_json,
                        NeedleStreamCallback callback,
                        void            *user_data);

/* ── Contrastive retrieval ────────────────────────────────────────────────── */

/**
 * Return the contrastive embedding dimension.
 *
 * Returns 0 if the model was loaded without a contrastive head
 * (i.e. the SafeTensors file does not contain contrastive_proj_kernel).
 */
size_t needle_contrastive_dim(NeedleHandle *handle);

/**
 * Encode text into a L2-normalized contrastive embedding.
 *
 * Both query and tool description embeddings are L2-normalized, so the
 * similarity score is just the dot product:
 *   score = sum(q_emb[i] * t_emb[i]) for i in 0..dim
 *
 * @param handle  Engine handle.
 * @param text    Null-terminated UTF-8 input string to encode.
 * @param out     Caller-allocated float32 buffer of at least needle_contrastive_dim(handle) elements.
 * @param dim     Size of `out` (must equal needle_contrastive_dim(handle)).
 * @return true on success, false if no contrastive head or dim mismatch.
 */
bool needle_encode_contrastive(NeedleHandle *handle,
                               const char   *text,
                               float        *out,
                               size_t        dim);

/**
 * Rank tool descriptions by contrastive similarity to a query.
 *
 * Encodes `query` and each `tool_descs[i]` with the contrastive projection head,
 * computes dot-product similarity (both embeddings are L2-normalized = cosine
 * similarity), and writes the top-k results into caller-supplied buffers sorted
 * by descending score.
 *
 * @param handle       Engine handle.
 * @param query        Null-terminated UTF-8 query string.
 * @param tool_descs   Array of `n_tools` null-terminated UTF-8 description strings.
 * @param n_tools      Number of tool descriptions.
 * @param top_k        Maximum number of results to return.
 * @param out_indices  Caller-allocated buffer of at least `top_k` size_t values.
 * @param out_scores   Caller-allocated buffer of at least `top_k` float values.
 * @return Number of results written (min(top_k, n_tools)), or 0 if no contrastive head.
 */
size_t needle_retrieve_tools(NeedleHandle  *handle,
                             const char    *query,
                             const char   **tool_descs,
                             size_t         n_tools,
                             size_t         top_k,
                             size_t        *out_indices,
                             float         *out_scores);

/* ── Memory management ────────────────────────────────────────────────────── */

/**
 * Free a string returned by needle_run or needle_run_stream.
 * Safe to call with NULL.
 */
void needle_free_str(char *s);

/**
 * Free a NeedleHandle returned by needle_load or needle_load_bytes.
 * Safe to call with NULL.
 */
void needle_free(NeedleHandle *handle);

/* ── Error reporting ──────────────────────────────────────────────────────── */

/**
 * Return the last error message as a null-terminated C string, or NULL if none.
 *
 * The returned pointer is valid until the next call to any needle_* function
 * on the current thread. Do NOT free this pointer.
 */
const char *needle_last_error(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NEEDLE_H */

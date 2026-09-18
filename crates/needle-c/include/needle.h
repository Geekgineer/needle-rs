/**
 * needle.h — C ABI for the needle-rs inference engine.
 *
 * Two model generations, two independent surfaces in one library:
 *
 *   needle_*     Needle v1 — .safetensors weights + a separate vocab file.
 *   needle_v2_*  Needle v2 — one .cact container (weights, geometry, tokenizer).
 *
 * They share only needle_free_str() and needle_last_error(). Handles are not
 * interchangeable: free a v2 handle with needle_v2_free(), never needle_free().
 *
 * Usage (compile against the shared library):
 *   gcc -o my_program my_program.c -lneedle_c -L./target/release
 *
 * Python ctypes example (Needle v2):
 *   import ctypes
 *   lib = ctypes.CDLL("./target/release/libneedle_c.so")
 *   lib.needle_v2_load.restype  = ctypes.c_void_p
 *   lib.needle_v2_load.argtypes = [ctypes.c_char_p]
 *   h = lib.needle_v2_load(b"weights/needle2.cact")
 *   # Declare returned strings as c_void_p, NOT c_char_p: ctypes converts a
 *   # c_char_p result into a Python bytes object and drops the original
 *   # pointer, so needle_free_str() would be handed Python-owned memory.
 *   lib.needle_v2_run.restype  = ctypes.c_void_p
 *   lib.needle_v2_run.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
 *   ptr = lib.needle_v2_run(h, b"What's the weather in Paris?", b'[{"name":"get_weather",...}]')
 *   print(ctypes.cast(ptr, ctypes.c_char_p).value.decode())
 *   lib.needle_free_str(ptr)
 *   lib.needle_v2_free(h)
 *
 * See examples/python-via-cffi/infer.py for a complete program covering both.
 */

#ifndef NEEDLE_H
#define NEEDLE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle returned by needle_load / needle_load_bytes (Needle v1). */
typedef struct NeedleHandle NeedleHandle;

/** Opaque handle returned by needle_v2_load / needle_v2_load_bytes (Needle v2). */
typedef struct NeedleV2Handle NeedleV2Handle;

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

/* ── Needle v1: loading ─────────────────────────────────────────────────── */

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

/* ── Needle v1: inference ───────────────────────────────────────────────── */

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

/* ── Needle v1: contrastive retrieval ───────────────────────────────────── */

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

/* ── Needle v2: loading ─────────────────────────────────────────────────── */

/**
 * Load a Needle v2 model from a `.cact` container on disk.
 *
 * The container carries weights, geometry and the tokenizer, so unlike
 * needle_load() there is no vocabulary argument.
 *
 * @param cact_path  Path to the .cact file (e.g. "weights/needle2.cact").
 * @return Opaque handle on success, NULL on failure.
 *         Call needle_last_error() to retrieve the error message.
 *         Caller must free the handle with needle_v2_free().
 */
NeedleV2Handle *needle_v2_load(const char *cact_path);

/**
 * Load a Needle v2 model from an in-memory `.cact` buffer.
 *
 * @param data  Pointer to raw .cact file bytes.
 * @param len   Length of data in bytes.
 * @return Opaque handle on success, NULL on failure.
 *         Caller must free the handle with needle_v2_free().
 */
NeedleV2Handle *needle_v2_load_bytes(const uint8_t *data, size_t len);

/* ── Needle v2: inference ───────────────────────────────────────────────── */

/**
 * Run single-example inference.
 *
 * @param handle     Engine handle from needle_v2_load / needle_v2_load_bytes.
 * @param query      Null-terminated UTF-8 query string.
 * @param tools_json Null-terminated UTF-8 JSON array of tool definitions.
 *                   Pretty-printed schemas are compacted internally, so
 *                   whitespace in the input does not change the result.
 * @return Heap-allocated null-terminated UTF-8 model output, including any
 *         <tool_call> markers. NULL on error; free with needle_free_str().
 */
char *needle_v2_run(NeedleV2Handle *handle, const char *query, const char *tools_json);

/**
 * As needle_v2_run, but returns only the <tool_call> payload.
 *
 * @return Heap-allocated JSON payload such as
 *         [{"name":"get_weather","arguments":{"city":"Paris"}}], or "[]" when
 *         no declared tool fits the query — a deliberate abstention, not an
 *         error. Returns NULL, with no error set, only when the output carried
 *         no <tool_call> markers at all, which is a degenerate generation.
 *         Free with needle_free_str().
 */
char *needle_v2_run_json(NeedleV2Handle *handle, const char *query, const char *tools_json);

/**
 * Generation with explicit settings.
 *
 * @param max_new_tokens Token budget for the completion.
 * @param temperature    <= 0 is greedy; > 0 samples.
 * @param seed           PRNG seed, used only when temperature > 0.
 * @param constrain      Non-zero restricts the tool-call payload to the
 *                       declared schema (valid names and argument keys only).
 * @return Same semantics as needle_v2_run. Free with needle_free_str().
 */
char *needle_v2_generate(NeedleV2Handle *handle,
                         const char     *query,
                         const char     *tools_json,
                         size_t          max_new_tokens,
                         float           temperature,
                         uint64_t        seed,
                         int             constrain);

/**
 * Run inference with a per-token streaming callback.
 *
 * Fires `callback` for each generated token, then returns the full output.
 * The same NeedleStreamCallback type is used as for the v1 surface.
 *
 * @param callback  Function called for each token (may be NULL to skip streaming).
 * @param user_data Passed through unchanged to each callback invocation.
 * @return Same semantics as needle_v2_run. Free with needle_free_str().
 */
char *needle_v2_run_stream(NeedleV2Handle  *handle,
                           const char      *query,
                           const char      *tools_json,
                           NeedleStreamCallback callback,
                           void            *user_data);

/* ── Needle v2: probe heads ─────────────────────────────────────────────── */

/**
 * Write the probability that `completion` answers `(query, tools_json)`
 * correctly to `out`, in the range (0, 1).
 *
 * The confidence head scores a judgement already made: it is trained on the
 * formatted prompt followed by a completion. This call assembles that input for
 * you, so pass the string needle_v2_run() returned for the run being judged.
 * Use it to gate execution — act above a threshold you pick, escalate below it.
 * This head has no v1 equivalent.
 *
 * @param completion  Model output being judged (e.g. from needle_v2_run).
 * @param out         Caller-allocated writable float.
 * @return true on success, false if the container has no confidence head.
 */
bool needle_v2_confidence_for(NeedleV2Handle *handle,
                              const char     *query,
                              const char     *tools_json,
                              const char     *completion,
                              float          *out);

/**
 * Write the raw confidence logit for an arbitrary string to `out`.
 *
 * Higher is more confident; apply a sigmoid for a probability:
 *   p = 1 / (1 + exp(-logit))
 *
 * This is the primitive. Because the head scores a prompt-plus-completion
 * pair, passing a bare query reads near zero however answerable it is — prefer
 * needle_v2_confidence_for() unless you are assembling the prompt yourself.
 *
 * @param out  Caller-allocated writable float.
 * @return true on success, false if the container has no confidence head.
 */
bool needle_v2_confidence(NeedleV2Handle *handle, const char *text, float *out);

/**
 * Return the contrastive embedding dimension, or 0 if the container has no
 * contrastive head.
 */
size_t needle_v2_contrastive_dim(NeedleV2Handle *handle);

/**
 * Encode text into an L2-normalized contrastive embedding.
 *
 * Both sides are L2-normalized, so similarity is the plain dot product.
 *
 * @param out  Caller-allocated float32 buffer of at least
 *             needle_v2_contrastive_dim(handle) elements.
 * @param dim  Size of `out` (must equal needle_v2_contrastive_dim(handle)).
 * @return true on success, false if no contrastive head or dim mismatch.
 */
bool needle_v2_encode_contrastive(NeedleV2Handle *handle,
                                  const char     *text,
                                  float          *out,
                                  size_t          dim);

/**
 * Rank tool descriptions by contrastive similarity to a query.
 *
 * Same contract as needle_retrieve_tools, on the v2 head.
 *
 * @param out_indices  Caller-allocated buffer of at least `top_k` size_t values.
 * @param out_scores   Caller-allocated buffer of at least `top_k` float values.
 * @return Number of results written (min(top_k, n_tools)), or 0 if no
 *         contrastive head.
 */
size_t needle_v2_retrieve_tools(NeedleV2Handle *handle,
                                const char     *query,
                                const char    **tool_descs,
                                size_t          n_tools,
                                size_t          top_k,
                                size_t         *out_indices,
                                float          *out_scores);

/**
 * Free a NeedleV2Handle returned by needle_v2_load / needle_v2_load_bytes.
 * Safe to call with NULL. Do NOT pass a v2 handle to needle_free().
 */
void needle_v2_free(NeedleV2Handle *handle);

/* ── Memory management ──────────────────────────────────────────────────── */

/**
 * Free a string returned by any needle_* or needle_v2_* call that returns
 * char* (shared by all three surfaces). Safe to call with NULL.
 */
void needle_free_str(char *s);

/**
 * Free a NeedleHandle returned by needle_load or needle_load_bytes.
 * Safe to call with NULL. For v2 handles use needle_v2_free().
 */
void needle_free(NeedleHandle *handle);

/* ── Needle 3 ───────────────────────────────────────────────────────────── */

/**
 * Opaque handle to a loaded Needle 3 engine.
 *
 * Needle 3 ships as a `.cact` container like v2, but the two are different
 * formats with different headers. The container states its generation in its
 * first word, so load the one that matches; a v2 container handed to
 * needle_v3_load() fails rather than being misread.
 */
typedef struct NeedleV3Handle NeedleV3Handle;

/** Load Needle 3 from a .cact file. NULL on failure; free with needle_v3_free(). */
NeedleV3Handle *needle_v3_load(const char *cact_path);

/**
 * Load the N-block ladder rung of a Needle 3 container.
 *
 * Every depth from 2 blocks up is a trained subnetwork; one file serves them
 * all, and a shallower rung costs proportionally less key/value cache. Quality
 * falls sharply at the bottom: 2 and 4 blocks do not produce usable tool calls
 * on the shipped weights, 6 upward do. NULL on failure.
 */
NeedleV3Handle *needle_v3_load_with_depth(const char *cact_path, size_t layers);

/** How many blocks a handle is running; 0 for NULL. */
size_t needle_v3_num_layers(NeedleV3Handle *handle);

/** Load Needle 3 from bytes already in memory. */
NeedleV3Handle *needle_v3_load_bytes(const unsigned char *data, size_t len);

/**
 * Full completion, reasoning included. Free with needle_free_str().
 *
 * Needle 3 usually answers with a <think> block before the call, which v2 did
 * not. Use needle_v3_run_json() if you only want the payload.
 */
char *needle_v3_run(NeedleV3Handle *handle, const char *query, const char *tools_json);

/**
 * Tool-call payload only. Free with needle_free_str().
 *
 * Returns "[]" when the model considered the tools and declined — a decision,
 * not a failure — and "" when it emitted no <tool_call> markers at all.
 * Collapsing the two turns a considered "no" into an error.
 */
char *needle_v3_run_json(NeedleV3Handle *handle, const char *query, const char *tools_json);

/**
 * The chain-of-thought inside a completion, or NULL if there is none.
 * Free with needle_free_str().
 */
char *needle_v3_reasoning(NeedleV3Handle *handle, const char *text);

/**
 * Generate with explicit settings. max_new_tokens == 0 uses the default.
 * `constrain` restricts the tool-call payload to the declared schema.
 */
char *needle_v3_generate(NeedleV3Handle *handle,
                         const char    *query,
                         const char    *tools_json,
                         size_t         max_new_tokens,
                         float          temperature,
                         uint64_t       seed,
                         bool           constrain,
                         bool           kv_int8);

/**
 * Generate, invoking cb(piece, userdata) with each decoded delta.
 *
 * The callback receives decoded text, not raw tokenizer pieces: concatenating
 * every delta reproduces the returned string exactly.
 */
char *needle_v3_run_stream(NeedleV3Handle *handle,
                           const char     *query,
                           const char     *tools_json,
                           void (*cb)(const char *piece, void *userdata),
                           void           *userdata);

/** Whether this container carries a confidence head. */
bool needle_v3_has_confidence(NeedleV3Handle *handle);

/**
 * How confident the model is in a completion it produced.
 *
 * PASS THE COMPLETION, NOT THE QUERY. The head scores a finished judgement.
 * On the shipped checkpoint a correct call scores 0.93 and a wrong one 0.26 —
 * but a bare query scores 0.80, which looks like a confident answer and is
 * not one. Needle 2 collapsed to near zero on a bare query, so that misuse
 * announced itself; Needle 3's does not.
 *
 * Writes a probability in (0, 1) to *out and returns true.
 */
bool needle_v3_confidence_for(NeedleV3Handle *handle,
                              const char     *query,
                              const char     *tools_json,
                              const char     *completion,
                              float          *out);

/**
 * Key/value cache bytes for a session of seq_len positions.
 *
 * Exposed because the caller usually owns the memory budget. Counts the cache
 * only; the packed weights are a separate fixed cost.
 */
size_t needle_v3_kv_bytes(NeedleV3Handle *handle, size_t seq_len, bool kv_int8);

/** Context limit in tokens. */
size_t needle_v3_max_seq_len(NeedleV3Handle *handle);

/** Release a Needle 3 handle. Safe to call with NULL. */
void needle_v3_free(NeedleV3Handle *handle);

/*
 * Deliberately absent: needle_v3_retrieve_tools and
 * needle_v3_encode_contrastive. Needle 3 exports a confidence head and nothing
 * else, so they would fail on every call. A missing symbol is a compile error
 * at the call site; a present one that always fails is a runtime mystery.
 */

/* ── Error reporting ────────────────────────────────────────────────────── */

/**
 * Return the last error message as a null-terminated C string, or NULL if none.
 * Shared by all three surfaces.
 *
 * The returned pointer is valid until the next call to any needle_* function
 * on the current thread. Do NOT free this pointer.
 */
const char *needle_last_error(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NEEDLE_H */

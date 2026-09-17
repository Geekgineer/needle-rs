#include "needle.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    NeedleV3Handle *h = needle_v3_load("weights/needle3.cact");
    if (!h) { printf("load failed: %s\n", needle_last_error()); return 1; }

    const char *tools =
      "[{\"name\":\"get_weather\",\"description\":\"Get current weather for a city\","
      "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},"
      "\"required\":[\"city\"]}}]";
    const char *q = "What's the weather in Paris?";

    int fails = 0;
    #define OK(c, m) do { if (c) printf("  ok   %s\n", m); else { printf("  FAIL %s\n", m); fails++; } } while (0)

    printf("  max_seq_len = %zu, kv_bytes(512) = %.1f MB\n",
           needle_v3_max_seq_len(h),
           needle_v3_kv_bytes(h, 512) / 1024.0 / 1024.0);
    OK(needle_v3_max_seq_len(h) == 8192, "max_seq_len reports the context limit");
    OK(needle_v3_has_confidence(h), "confidence head present");

    char *text = needle_v3_run(h, q, tools);
    OK(text && strstr(text, "get_weather"), "run produces a get_weather call");
    printf("  run -> %s\n", text ? text : "(null)");

    char *reason = needle_v3_reasoning(h, text);
    printf("  reasoning -> %s\n", reason ? reason : "(none)");
    if (reason) needle_free_str(reason);

    char *payload = needle_v3_run_json(h, q, tools);
    OK(payload && payload[0] == '[', "run_json returns the payload");
    printf("  run_json -> %s\n", payload ? payload : "(null)");

    float p_right = 0.0f, p_bare = 0.0f;
    OK(needle_v3_confidence_for(h, q, tools, text, &p_right), "confidence_for on the completion");
    OK(needle_v3_confidence_for(h, q, tools, "", &p_bare), "confidence_for on a bare query");
    printf("  confidence: completion %.4f, bare query %.4f\n", p_right, p_bare);
    OK(p_right > p_bare, "the real completion outscores a bare query");

    char *bound = needle_v3_generate(h, q, tools, 0, 0.0f, 0, true);
    OK(bound && strstr(bound, "get_weather"), "constrained generate still answers");
    if (bound) needle_free_str(bound);

    needle_free_str(text);
    needle_free_str(payload);
    needle_v3_free(h);

    /* Null handling must not crash. */
    needle_v3_free(NULL);
    OK(needle_v3_load("nope.cact") == NULL, "a missing file returns NULL");
    OK(needle_v3_kv_bytes(NULL, 1) == 0, "a null handle reports zero");

    printf("\n%s\n", fails ? "FAILED" : "all C ABI checks passed");
    return fails ? 1 : 0;
}

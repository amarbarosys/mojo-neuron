// A5 oracle (docs/prereg-a5.md): run a qwen35moe GGUF on CPU over a token id file and print llama.cpp's own router tensors:
// ffn_moe_topk (the selected expert ids), ffn_moe_weights (softmax of the picked experts) and ffn_moe_weights_norm (after the top-k renormalisation, when the graph has it).
// usage: moe-topk-oracle MODEL.gguf IDS.txt > out.txt      lines: "<tensor> <layer> <token> <e0> <e1> ... <e7>" (ids as integers, weights as floats)
// llama-eval-callback cannot be the oracle: it elides the middle of an 8-wide row.
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>

static bool cb(struct ggml_tensor * t, bool ask, void *) {
    const char * n = t->name;
    bool want = !strncmp(n, "ffn_moe_topk-", 13) || !strncmp(n, "ffn_moe_weights-", 16) || !strncmp(n, "ffn_moe_weights_norm-", 21);
    if (ask) return want;
    if (!want) return true;
    const int layer = atoi(strrchr(n, '-') + 1);
    std::string base(n, strrchr(n, '-') - n);
    // strides matter: ggml_argsort_top_k returns a VIEW whose row stride is n_expert, not n_expert_used (reading j*k + e is right for token 0 only)
    const bool w = t->ne[0] == 1;                                       // weights are [1, k, n_tokens], ids [k, n_tokens]
    const int64_t k = w ? t->ne[1] : t->ne[0];
    const int64_t nt = w ? t->ne[2] : t->ne[1];
    const size_t se = w ? t->nb[1] : t->nb[0], st = w ? t->nb[2] : t->nb[1];
    std::vector<unsigned char> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
    for (int64_t j = 0; j < nt; j++) {
        printf("%s %d %lld", base.c_str(), layer, (long long) j);
        for (int64_t e = 0; e < k; e++) {
            const unsigned char * p = buf.data() + j * st + e * se;
            if (t->type == GGML_TYPE_I32) printf(" %d", *(const int *) p);
            else printf(" %.6f", *(const float *) p);
        }
        printf("\n");
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: moe-topk-oracle MODEL.gguf IDS.txt\n"); return 2; }
    std::vector<llama_token> ids; { std::ifstream f(argv[2]); int x; while (f >> x) ids.push_back(x); }
    if (ids.empty()) { fprintf(stderr, "no token ids in %s\n", argv[2]); return 2; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params(); cp.n_ctx = 256; cp.n_batch = 256; cp.cb_eval = cb; cp.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }
    if (llama_decode(ctx, llama_batch_get_one(ids.data(), (int) ids.size()))) { fprintf(stderr, "decode failed\n"); return 1; }
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}

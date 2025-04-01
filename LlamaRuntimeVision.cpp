#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "clip.h"
#include "stb_image.h"
#include "llama.h"
#include "ggml.h"
#include "console.h"

#include <vector>
#include <limits.h>
#include <inttypes.h>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

static bool g_is_generating = false;

/**
 * Please note that this is NOT a production-ready stuff.
 * It is a playground for trying Gemma 3 vision capabilities.
 * For contributors: please keep this code simple and easy to understand.
 */

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG(
        "Experimental CLI for using Gemma 3 vision model\n\n"
        "Usage: %s [options] -m <model> --mmproj <mmproj> --image <image> -p <prompt>\n\n"
        "  -m and --mmproj are required\n"
        "  --image and -p are optional, if NOT provided, the CLI will run in chat mode\n",
        argv[0]
        );
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (g_is_generating) {
            g_is_generating = false;
        } else {
            console::cleanup();
            LOG("\nInterrupted by user\n");
            _exit(130);
        }
    }
}
#endif

struct gemma3_context {
    struct clip_ctx    * ctx_clip = NULL;
    common_init_result   llama_init;

    llama_model       * model;
    llama_context     * lctx;
    const llama_vocab * vocab;
    llama_batch         batch;

    int n_threads    = 1;
    llama_pos n_past = 0;

    gemma3_context(common_params & params) : llama_init(common_init_from_params(params)) {
        model = llama_init.model.get();
        lctx = llama_init.context.get();
        vocab = llama_model_get_vocab(model);
        n_threads = params.cpuparams.n_threads;
        batch = llama_batch_init(params.n_batch, 0, 1);
        init_clip_model(params);
    }

    void init_clip_model(common_params & params) {
        const char * clip_path = params.mmproj.c_str();
        ctx_clip = clip_model_load(clip_path, params.verbosity > 1);
    }

    ~gemma3_context() {
        clip_free(ctx_clip);
    }
};

struct decode_embd_batch {
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch batch;
    decode_embd_batch(float * embd, int32_t n_tokens, llama_pos pos_0, llama_seq_id seq_id) {
        pos     .resize(n_tokens);
        n_seq_id.resize(n_tokens);
        seq_ids .resize(n_tokens + 1);
        logits  .resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0] = seq_id;
        seq_ids [n_tokens] = nullptr;
        batch = {
            /*n_tokens       =*/ n_tokens,
            /*tokens         =*/ nullptr,
            /*embd           =*/ embd,
            /*pos            =*/ pos.data(),
            /*n_seq_id       =*/ n_seq_id.data(),
            /*seq_id         =*/ seq_ids.data(),
            /*logits         =*/ logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            batch.pos     [i] = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }
};

static int eval_text(gemma3_context & ctx, std::string input, bool logits_last = false) {
    llama_tokens tokens = common_tokenize(ctx.lctx, input, false, true);
    common_batch_clear(ctx.batch);
    for (llama_token & t : tokens) {
        common_batch_add(ctx.batch, t, ctx.n_past++, {0}, false);
    }
    if (logits_last) {
        ctx.batch.logits[ctx.batch.n_tokens - 1] = true;
    }
    // LOG("eval_text (n_tokens = %d): %s\n", (int)tokens.size(), input.c_str());
    if (llama_decode(ctx.lctx, ctx.batch)) {
        LOG_ERR("Failed to decode text\n");
        return 1;
    }
    return 0;
}

static int eval_image(gemma3_context & ctx, std::string & fname) {
    std::vector<float> image_embd_v;
    int n_embd = llama_model_n_embd(ctx.model);
    int n_tokens = 256;
    image_embd_v.resize(n_tokens * n_embd);

    bool ok;
    struct clip_image_u8 * img_u8 = clip_image_u8_init();
    ok = clip_image_load_from_file(fname.c_str(), img_u8);
    if (!ok) {
        LOG_ERR("Unable to load image %s\n", fname.c_str());
        clip_image_u8_free(img_u8);
        return 2; // non-fatal error
    }

    clip_image_f32_batch batch_f32;
    ok = clip_image_preprocess(ctx.ctx_clip, img_u8, &batch_f32);
    if (!ok) {
        LOG_ERR("Unable to preprocess image\n");
        clip_image_f32_batch_free(&batch_f32);
        clip_image_u8_free(img_u8);
        return 1;
    }

    int64_t t0 = ggml_time_ms();
    LOG("Encoding image %s\n", fname.c_str());
    ok = clip_image_batch_encode(ctx.ctx_clip, ctx.n_threads, &batch_f32, image_embd_v.data());
    if (!ok) {
        LOG_ERR("Unable to encode image\n");
        clip_image_f32_batch_free(&batch_f32);
        clip_image_u8_free(img_u8);
        return 1;
    }
    LOG("Image encoded in %" PRId64 " ms\n", ggml_time_ms() - t0);

    clip_image_f32_batch_free(&batch_f32);
    clip_image_u8_free(img_u8);

    // decode image embeddings
    int64_t t1 = ggml_time_ms();
    eval_text(ctx, "<start_of_image>");
    llama_set_causal_attn(ctx.lctx, false);
    decode_embd_batch batch_img(image_embd_v.data(), n_tokens, ctx.n_past, 0);
    if (llama_decode(ctx.lctx, batch_img.batch)) {
        LOG_ERR("failed to decode image\n");
        return 1;
    }
    ctx.n_past += n_tokens;
    llama_set_causal_attn(ctx.lctx, true);
    eval_text(ctx, "<end_of_image>");
    LOG("Image decoded in %" PRId64 " ms\n", ggml_time_ms() - t1);
    return 0;
}

static int generate_response(gemma3_context & ctx, common_sampler * smpl, int n_predict, void (*callback)(const char*, void *userData), void *userData) {
    for (int i = 0; i < n_predict; i++) {
        if (i > n_predict || !g_is_generating) {
            printf("\n");
            break;
        }

        llama_token token_id = common_sampler_sample(smpl, ctx.lctx, -1);
        common_sampler_accept(smpl, token_id, true);

        if (llama_vocab_is_eog(ctx.vocab, token_id)) {
            printf("\n");
            break; // end of generation
        }

        std::string piece = common_token_to_piece(ctx.lctx, token_id).c_str();
        printf("%s", piece.c_str());
        if(callback)
            callback(piece.c_str(), userData);

        fflush(stdout);

        // eval the token
        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("failed to decode token\n");
            return 1;
        }
    }
    return 0;
}

#define MAIN
#ifdef MAIN

common_params params;
gemma3_context *ctx=nullptr;
struct common_sampler * smpl = nullptr;
llama_pos image_end_pos = 0;

bool hasVision(){
    return ctx != nullptr;
}




bool generateVision(int session_id, const std::string &line, void (*callback)(const char*, void *userData), void *userData)
{
    //params.prompt = line;

    int n_predict = params.n_predict < 0 ? INT_MAX : params.n_predict;

    if (line.empty()) {
        return false;
    }
    if (line == "/quit" || line == "/exit") {
        return true;
    }
    if (line == "/info") {
        int n_used = llama_get_kv_cache_used_cells(ctx->lctx);
        int n_max = llama_n_ctx(ctx->lctx);

        // Log the current state
        LOG("Current KV cache usage: %d / %d tokens\n", n_used, n_max);
        std::string msg = "Current KV cache usage: " + std::to_string(n_used) + " / " + std::to_string(n_max) + " tokens\n";

        if(callback)
            callback(msg.c_str(), userData);
    }

    if (line == "/clear") {
        int n_used = llama_get_kv_cache_used_cells(ctx->lctx);
        int n_max = llama_n_ctx(ctx->lctx);

        // Log the current state
        LOG("Current KV cache usage: %d / %d tokens\n", n_used, n_max);
        std::string msg = "Current KV cache usage: " + std::to_string(n_used) + " / " + std::to_string(n_max) + " tokens\n";

        if(callback)
            callback(msg.c_str(), userData);

        // Check current KV cache usage

        // Log the current state
        LOG("Current KV cache usage: %d / %d tokens\n", n_used, n_max);

        if (image_end_pos > 0 && image_end_pos < n_max - 100) { // Ensure we have room
            // Option 1: Keep BOS and image, remove everything else
            llama_kv_cache_clear(ctx->lctx); // Clear entire cache

            // Reload critical parts
            ctx->n_past = 0;

            // Reload BOS
            eval_text(*ctx, "<bos>");

            // Reload user turn
            eval_text(*ctx, "<start_of_turn>user\n");

            // Reload the initial images
            for (auto& fname : params.image) {
                if (eval_image(*ctx, fname)) {
                    LOG("Warning: Failed to reload image\n");
                }
            }

            LOG("Context reset: BOS and images reloaded (%d tokens)\n", ctx->n_past);
        } else {
            // Option 2: Complete reset if context is nearly full
            llama_kv_cache_clear(ctx->lctx);
            ctx->n_past = 0;

            // Restart with just BOS
            eval_text(*ctx, "<bos>");
            eval_text(*ctx, "<start_of_turn>user\n");

            LOG("Full context reset (context was too full)\n");
        }

        return true;
    }









    g_is_generating = true;
    if (line.find("/image") == 0) {
        std::string image = line.substr(7);
        int res = eval_image(*ctx, image);
        if (res == 2) {
            return false; // image not found
        }
        if (res) {
            return true;
        }
        return true;
    }
    if (eval_text(*ctx, line + "<end_of_turn><start_of_turn>model\n", true)) {
        return true;
    }
    if (generate_response(*ctx, smpl, n_predict, callback, userData)) {
        return true;
    }
    if (eval_text(*ctx, "<end_of_turn><start_of_turn>user\n")) {
        return true;
    }
}

int main_vision(const char *prompt, const char *image) {

    ggml_time_init();


    params.sampling.temp = 0.2; // lower temp by default for better quality

    /*if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LLAVA, show_additional_info)) {
        return 1;
    }*/

    params.n_ctx - 4096*2;
    params.prompt = prompt; // "Please analyse this image in greate details! Thank you, much apreciated!";
    params.model = "/Users/andreascarlen/.cache/EchoLlama/models/gemma-3-12b-it-q4_0.gguf";
    params.mmproj = "/Users/andreascarlen/.cache/EchoLlama/models/mmproj-google_gemma-3-12b-it-f16.gguf";        // path to multimodal projector                                         // NOLINT
    params.image.push_back(image/*"/Users/andreascarlen/Documents/Screenshot 2024-03-22 at 12.09.51.png"*/); // path to image file(s)

    common_init();

    if(!ctx)
        ctx = new gemma3_context(params);
    //gemma3_context ctx(params);
    printf("%s: %s\n", __func__, params.model.c_str());

    bool is_single_turn = !params.prompt.empty() && !params.image.empty();

    smpl = common_sampler_init(ctx->model, params.sampling);
    int n_predict = params.n_predict < 0 ? INT_MAX : params.n_predict;

    // ctrl+C handling
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    if (eval_text(*ctx, "<bos>")) {
        return 1;
    }

    if (is_single_turn) {
        g_is_generating = true;
        if (eval_text(*ctx, "<start_of_turn>user\n")) {
            return 1;
        }
        for (auto & fname : params.image) {
            if (eval_image(*ctx, fname)) {
                return 1;
            }
        }

        // Store the position after images are processed
        image_end_pos = ctx->n_past;

        if (eval_text(*ctx, params.prompt + "<end_of_turn><start_of_turn>model\n", true)) {
            return 1;
        }
        if (generate_response(*ctx, smpl, n_predict, nullptr, nullptr)) {
            return 1;
        }

    }

    return 0;
}
#endif






///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
/// ///////////////////////////////////////////////////////////////////////
///
///
///
///
///
///
///
///
///
///
///
/// ///////////////////////////////////////////////////////////////////////
/// ///////////////////////////////////////////////////////////////////////
/// ///////////////////////////////////////////////////////////////////////
///
///
///
///
///
///
///
///
///
/// ///////////////////////////////////////////////////////////////////////
/// ///////////////////////////////////////////////////////////////////////
/// ///////////////////////////////////////////////////////////////////////



#include "LlamaRuntimeVision.h"






// Implementation of the LlamaRuntimeVision class methods
LlamaRuntimeVision::LlamaRuntimeVision() : params(nullptr), ctx(nullptr), smpl(nullptr), image_end_pos(0), is_initialized(false) {
    // Initialize ggml timing
    ggml_time_init();

    // Allocate params
    params = new common_params();
    if (params) {
        // Set some default values
        params->sampling.temp = 0.2f; // lower temp by default for better quality
        params->n_ctx = 8192;
        params->n_batch = 512;
        params->n_predict = -1; // default to unlimited
    }
}

LlamaRuntimeVision::~LlamaRuntimeVision() {
    // Clean up resources
    if (smpl) {
        common_sampler_free(smpl);
        smpl = nullptr;
    }

    if (ctx) {
        delete ctx;
        ctx = nullptr;
    }

    if (params) {
        delete params;
        params = nullptr;
    }
}

bool LlamaRuntimeVision::initialize(const std::string& model_path,
                                    const std::string& mmproj_path,
                                    float temperature,
                                    int context_size) {
    if (!params) {
        LOG_ERR("Parameters not initialized\n");
        return false;
    }

    // Set up parameters
    params->model = model_path;
    params->mmproj = mmproj_path;
    params->sampling.temp = temperature;
    params->n_ctx = context_size;

    // Initialize the common framework
    common_init();

    // Create context
    try {
        ctx = new gemma3_context(*params);

        // Initialize sampler
        smpl = common_sampler_init(ctx->model, params->sampling);

// Set up SIGINT handler
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

        // Initialize the context with BOS
        if (eval_text(*ctx, "<bos>")) {
            LOG_ERR("Failed to initialize context with BOS\n");
            return false;
        }

        is_initialized = true;
        return true;
    } catch (const std::exception& e) {
        LOG_ERR("Exception during initialization: %s\n", e.what());
        return false;
    } catch (...) {
        LOG_ERR("Unknown exception during initialization\n");
        return false;
    }
}

bool LlamaRuntimeVision::hasVision() {
    return ctx != nullptr && ctx->ctx_clip != nullptr;
}

bool LlamaRuntimeVision::processImageAndGenerate(const std::string& prompt,
                                                 const std::string& image_path,
                                                 void (*callback)(const char*, void* userData),
                                                 void* userData) {
    if (!is_initialized || !ctx || !smpl) {
        LOG_ERR("System not initialized\n");
        return false;
    }

    // Store the image path in params
    params->image.clear();
    params->image.push_back(image_path);
    params->prompt = prompt;

    int n_predict = params->n_predict < 0 ? INT_MAX : params->n_predict;

    g_is_generating = true;

    // Start user turn
    if (eval_text(*ctx, "<start_of_turn>user\n")) {
        LOG_ERR("Failed to initialize user turn\n");
        return false;
    }

    // Process the image
    for (auto& fname : params->image) {
        if (eval_image(*ctx, fname)) {
            LOG_ERR("Failed to process image: %s\n", fname.c_str());
            return false;
        }
    }

    // Store the position after images are processed
    image_end_pos = ctx->n_past;

    // Process prompt and end user turn, start model turn
    if (eval_text(*ctx, prompt + "<end_of_turn><start_of_turn>model\n", true)) {
        LOG_ERR("Failed to process prompt\n");
        return false;
    }

    // Generate response
    if (generate_response(*ctx, smpl, n_predict, callback, userData)) {
        LOG_ERR("Failed to generate response\n");
        return false;
    }

    // End model turn and start new user turn
    if (eval_text(*ctx, "<end_of_turn><start_of_turn>user\n")) {
        LOG_ERR("Failed to end model turn\n");
        return false;
    }

    g_is_generating = false;
    return true;
}

bool LlamaRuntimeVision::generateResponse(const std::string& prompt,
                                          void (*callback)(const char*, void* userData),
                                          void* userData) {
    if (!is_initialized || !ctx || !smpl) {
        LOG_ERR("System not initialized\n");
        return false;
    }

    int n_predict = params->n_predict < 0 ? INT_MAX : params->n_predict;

    if (prompt.empty()) {
        LOG_ERR("Empty prompt\n");
        return false;
    }

    g_is_generating = true;

    // Process prompt and end user turn, start model turn
    if (eval_text(*ctx, prompt + "<end_of_turn><start_of_turn>model\n", true)) {
        LOG_ERR("Failed to process prompt\n");
        return false;
    }

    // Generate response
    if (generate_response(*ctx, smpl, n_predict, callback, userData)) {
        LOG_ERR("Failed to generate response\n");
        return false;
    }

    // End model turn and start new user turn
    if (eval_text(*ctx, "<end_of_turn><start_of_turn>user\n")) {
        LOG_ERR("Failed to end model turn\n");
        return false;
    }

    g_is_generating = false;
    return true;
}

bool LlamaRuntimeVision::clearContext(bool keepImages) {
    if (!ctx) {
        LOG_ERR("Context not initialized\n");
        return false;
    }

    int n_used = llama_get_kv_cache_used_cells(ctx->lctx);
    int n_max = llama_n_ctx(ctx->lctx);

    // Log the current state
    LOG("Current KV cache usage: %d / %d tokens\n", n_used, n_max);

    if (keepImages && image_end_pos > 0 && image_end_pos < n_max - 100) {
        // Option 1: Keep BOS and image, remove everything else
        llama_kv_cache_clear(ctx->lctx);

        // Reload critical parts
        ctx->n_past = 0;

        // Reload BOS
        eval_text(*ctx, "<bos>");

        // Reload user turn
        eval_text(*ctx, "<start_of_turn>user\n");

        // Reload the initial images
        for (auto& fname : params->image) {
            if (eval_image(*ctx, fname)) {
                LOG("Warning: Failed to reload image\n");
            }
        }

        LOG("Context reset: BOS and images reloaded (%d tokens)\n", ctx->n_past);
    } else {
        // Option 2: Complete reset if context is nearly full or image retention not requested
        llama_kv_cache_clear(ctx->lctx);
        ctx->n_past = 0;

        // Restart with just BOS
        eval_text(*ctx, "<bos>");
        eval_text(*ctx, "<start_of_turn>user\n");

        LOG("Full context reset (context was too full or image retention not requested)\n");
    }

    return true;
}

std::string LlamaRuntimeVision::getContextInfo() {
    if (!ctx) {
        return "Context not initialized";
    }

    int n_used = llama_get_kv_cache_used_cells(ctx->lctx);
    int n_max = llama_n_ctx(ctx->lctx);

    char buffer[256];
    snprintf(buffer, sizeof(buffer),
             "Current KV cache usage: %d / %d tokens\n"
             "Image end position: %d\n"
             "Model: %s\n"
             "CLIP model: %s\n",
             n_used, n_max, image_end_pos,
             params->model.c_str(), params->mmproj.c_str());

    return std::string(buffer);
}



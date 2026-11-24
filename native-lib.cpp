#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include "llama.h"

#define LOG_TAG "NovaLlama"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_nova_server_MainActivity_loadModel(JNIEnv* env, jobject, jstring model_path) {
    const char* path = env->GetStringUTFChars(model_path, nullptr);
    LOGI("Loading model from: %s", path);

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; 

    g_model = llama_load_model_from_file(path, mparams);
    if (!g_model) {
        LOGI("Failed to load model");
        env->ReleaseStringUTFChars(model_path, path);
        return JNI_FALSE;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1024; 
    cparams.n_threads = 4; 

    g_ctx = llama_new_context_with_model(g_model, cparams);
    if (!g_ctx) {
        LOGI("Failed to create context");
        llama_free_model(g_model);
        g_model = nullptr;
        env->ReleaseStringUTFChars(model_path, path);
        return JNI_FALSE;
    }

    env->ReleaseStringUTFChars(model_path, path);
    LOGI("Model loaded successfully");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nova_server_MainActivity_generateResponse(JNIEnv* env, jobject, jstring prompt) {
    if (!g_ctx || !g_model) {
        return env->NewStringUTF("خطأ: النموذج غير محمل. يرجى تحميل النموذج أولاً.");
    }

    const char* prompt_cstr = env->GetStringUTFChars(prompt, nullptr);
    LOGI("Prompt: %s", prompt_cstr);

    std::string full_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n";
    full_prompt += prompt_cstr;
    full_prompt += "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<llama_token> tokens(full_prompt.length() + 4);
    int n_tokens = llama_tokenize(g_model, full_prompt.c_str(), full_prompt.length(), tokens.data(), tokens.size(), true, true);
    
    if (n_tokens < 0) {
        env->ReleaseStringUTFChars(prompt, prompt_cstr);
        return env->NewStringUTF("خطأ في تحويل النص إلى رموز.");
    }
    tokens.resize(n_tokens);

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

    if (llama_decode(g_ctx, batch)) {
        env->ReleaseStringUTFChars(prompt, prompt_cstr);
        return env->NewStringUTF("خطأ في معالجة النص.");
    }

    std::string response;
    int n_predict = 256; 
    for (int i = 0; i < n_predict; i++) {
        llama_token id = llama_sample_token_greedy(g_ctx, nullptr);
        
        if (id == llama_token_eos(g_model)) break;

        char buf[128];
        int n = llama_token_to_piece(g_model, id, buf, sizeof(buf), 0, false);
        if (n > 0) {
            response.append(buf, n);
        }

        llama_batch next_batch = llama_batch_get_one(&id, 1);
        if (llama_decode(g_ctx, next_batch)) break;
    }

    env->ReleaseStringUTFChars(prompt, prompt_cstr);
    return env->NewStringUTF(response.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_nova_server_MainActivity_freeModel(JNIEnv*, jobject) {
    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_free_model(g_model); g_model = nullptr; }
}

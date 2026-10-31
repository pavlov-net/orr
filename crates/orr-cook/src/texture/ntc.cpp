// Compresses one texture set with LibNTC and hands back what the neural texture set class
// stores. LibNTC's entry points come in as function pointers from the library the cook
// loaded; the interfaces they return are called through their vtables.

#include <libntc/ntc.h>

#include <cstdio>
#include <cstring>
#include <vector>

static_assert(sizeof(NtcTextureSetConstants) == 336, "NtcTextureSetConstants changed size");

extern "C" {

struct OrrNtcApi {
    ntc::Status (*create_context)(ntc::IContext**, ntc::ContextParameters const&);
    void (*destroy_context)(ntc::IContext*);
    char const* (*last_error)();
    float (*loss_to_psnr)(float);
};

struct OrrNtcTexture {
    int first_channel;
    int channel_count;
    // 1: RGB channels are stored as sRGB.
    int srgb;
    // ntc::ChannelFormat for decompressed storage.
    int format;
};

struct OrrNtcInput {
    int width;
    int height;
    int mips;
    int channels;
    // Per mip, max(1, w >> m) x max(1, h >> m) texels of `channels` floats, rows tight.
    float const* const* mip_data;
    // Per channel, 1 where the network stores the channel as sRGB.
    unsigned char const* channel_srgb;
    int texture_count;
    OrrNtcTexture const* textures;
    float target_psnr;
    float max_bits_per_pixel;
    int training_steps;
    unsigned seed;
    int cuda_device;
};

struct OrrNtcResult {
    int latent_width;
    int latent_height;
    int latent_layers;
    int latent_mips;
    int latent_features;
    float bits_per_pixel;
    // Overall PSNR of the chosen run, dB.
    float psnr;
    int runs;
};

enum OrrNtcSinkKind {
    // index = mip, extra = layer; rows tight, 2 bytes per texel.
    ORR_NTC_LATENT = 0,
    // index = weight type; extra = bytes after conversion to the device layout.
    ORR_NTC_WEIGHTS = 1,
    // index = weight type.
    ORR_NTC_CONSTANTS = 2,
    // Mip 0 of the decompressed output: w x h texels of `channels` floats, color channels
    // sRGB-encoded where the input marks them.
    ORR_NTC_DECODED = 3,
};

typedef void (*OrrNtcSink)(void* user, int kind, int index, void const* data, size_t len,
                           unsigned long long extra);

int orr_ntc_compress(OrrNtcApi const* api, OrrNtcInput const* in, OrrNtcResult* out,
                     OrrNtcSink sink, void* user, char* error, size_t error_len);
}

namespace {

struct Failure {};

struct Run {
    OrrNtcApi const* api;
    char* error;
    size_t error_len;

    void check(ntc::Status s, char const* what) const {
        if (s != ntc::Status::Ok) {
            char const* detail = api->last_error ? api->last_error() : nullptr;
            std::snprintf(error, error_len, "%s: status %d: %s", what, int(s),
                          detail ? detail : "");
            throw Failure{};
        }
    }

    [[noreturn]] void fail(char const* what) const {
        std::snprintf(error, error_len, "%s", what);
        throw Failure{};
    }
};

struct Context {
    OrrNtcApi const* api;
    ntc::IContext* ptr = nullptr;
    ~Context() {
        if (ptr) api->destroy_context(ptr);
    }
};

struct TextureSet {
    ntc::IContext* context;
    ntc::ITextureSet* ptr = nullptr;
    ~TextureSet() {
        if (ptr) context->DestroyTextureSet(ptr);
    }
};

struct Session {
    ntc::IContext* context;
    ntc::IAdaptiveCompressionSession* ptr = nullptr;
    ~Session() {
        if (ptr) context->DestroyAdaptiveCompressionSession(ptr);
    }
};

struct Metadata {
    ntc::IContext* context;
    ntc::ITextureSetMetadata* ptr = nullptr;
    ~Metadata() {
        if (ptr) context->DestroyTextureSetMetadata(ptr);
    }
};

struct Stream {
    ntc::IContext* context;
    ntc::IStream* ptr = nullptr;
    ~Stream() {
        if (ptr) context->CloseMemory(ptr);
    }
};

int mip_extent(int size, int mip) {
    int e = size >> mip;
    return e > 0 ? e : 1;
}

void compress(Run const& run, OrrNtcInput const* in, OrrNtcResult* out, OrrNtcSink sink,
              void* user) {
    OrrNtcApi const* api = run.api;
    if (in->channels < 1 || in->channels > NTC_MAX_CHANNELS || in->mips < 1 ||
        in->mips > NTC_MAX_MIPS)
        run.fail("channel or mip count outside LibNTC's range");

    Context context{api};
    ntc::ContextParameters cp;
    cp.cudaDevice = in->cuda_device;
    cp.graphicsApi = ntc::GraphicsAPI::None;
    run.check(api->create_context(&context.ptr, cp), "CreateContext");
    ntc::IContext* ctx = context.ptr;

    ntc::TextureSetDesc desc;
    desc.width = in->width;
    desc.height = in->height;
    desc.channels = in->channels;
    desc.mips = in->mips;
    ntc::TextureSetFeatures features;
    features.stagingBytesPerPixel = int(sizeof(float)) * in->channels;
    features.separateRefOutData = true;
    TextureSet set{ctx};
    run.check(ctx->CreateTextureSet(desc, features, &set.ptr), "CreateTextureSet");

    for (int t = 0; t < in->texture_count; ++t) {
        OrrNtcTexture const& tex = in->textures[t];
        ntc::ITextureMetadata* meta = set.ptr->AddTexture();
        if (!meta) run.fail("AddTexture");
        char name[16];
        std::snprintf(name, sizeof name, "t%d", t);
        meta->SetName(name);
        run.check(meta->SetChannels(tex.first_channel, tex.channel_count), "SetChannels");
        meta->SetChannelFormat(ntc::ChannelFormat(tex.format));
        meta->SetRgbColorSpace(tex.srgb ? ntc::ColorSpace::sRGB : ntc::ColorSpace::Linear);
        meta->SetAlphaColorSpace(ntc::ColorSpace::Linear);
    }

    ntc::ColorSpace linear[NTC_MAX_CHANNELS];
    ntc::ColorSpace stored[NTC_MAX_CHANNELS];
    for (int c = 0; c < in->channels; ++c) {
        linear[c] = ntc::ColorSpace::Linear;
        stored[c] = in->channel_srgb[c] ? ntc::ColorSpace::sRGB : ntc::ColorSpace::Linear;
    }
    size_t const texel = sizeof(float) * size_t(in->channels);
    for (int m = 0; m < in->mips; ++m) {
        ntc::WriteChannelsParameters w;
        w.mipLevel = m;
        w.firstChannel = 0;
        w.numChannels = in->channels;
        w.pData = reinterpret_cast<unsigned char const*>(in->mip_data[m]);
        w.addressSpace = ntc::AddressSpace::Host;
        w.width = mip_extent(in->width, m);
        w.height = mip_extent(in->height, m);
        w.pixelStride = texel;
        w.rowPitch = texel * size_t(w.width);
        w.channelFormat = ntc::ChannelFormat::FLOAT32;
        w.srcColorSpaces = linear;
        w.dstColorSpaces = stored;
        run.check(set.ptr->WriteChannels(w), "WriteChannels");
    }

    // The archive compresses every blob itself, and latent footprints must be raw texels.
    ntc::LosslessCompressionSettings lossless;
    lossless.algorithm = ntc::CompressionType::None;
    lossless.compressBCModeBuffers = false;
    lossless.compressLatents = false;
    run.check(set.ptr->ConfigureLosslessCompression(lossless), "ConfigureLosslessCompression");

    ntc::CompressionSettings settings;
    settings.trainingSteps = in->training_steps;
    settings.randomSeed = in->seed;
    settings.stableTraining = true;

    Session session{ctx};
    run.check(ctx->CreateAdaptiveCompressionSession(&session.ptr),
              "CreateAdaptiveCompressionSession");
    run.check(session.ptr->Reset(in->target_psnr, in->max_bits_per_pixel), "Reset");
    std::vector<std::vector<unsigned char>> versions;
    std::vector<float> psnrs;
    while (!session.ptr->Finished()) {
        float bpp = 0.f;
        ntc::LatentShape shape;
        session.ptr->GetCurrentPreset(&bpp, &shape);
        run.check(set.ptr->SetLatentShape(shape), "SetLatentShape");
        run.check(set.ptr->BeginCompression(settings), "BeginCompression");
        ntc::CompressionStats stats;
        ntc::Status s;
        do {
            s = set.ptr->RunCompressionSteps(&stats);
        } while (s == ntc::Status::Incomplete);
        run.check(s, "RunCompressionSteps");
        run.check(set.ptr->FinalizeCompression(), "FinalizeCompression");
        ntc::DecompressionStats ds;
        run.check(set.ptr->Decompress(&ds, false), "Decompress");
        float psnr = api->loss_to_psnr(ds.overallLoss);
        size_t size = size_t(set.ptr->GetOutputStreamSize());
        std::vector<unsigned char> bytes(size);
        run.check(set.ptr->SaveToMemory(bytes.data(), &size), "SaveToMemory");
        bytes.resize(size);
        versions.push_back(std::move(bytes));
        psnrs.push_back(psnr);
        session.ptr->Next(psnr);
    }
    int const last = session.ptr->GetIndexOfFinalRun();
    if (last < 0 || size_t(last) >= versions.size()) run.fail("no final compression run");
    std::vector<unsigned char> const& file = versions[size_t(last)];
    ntc::LatentShape chosen;
    session.ptr->GetCurrentPreset(&out->bits_per_pixel, &chosen);
    out->psnr = psnrs[size_t(last)];
    out->runs = int(versions.size());

    run.check(set.ptr->LoadFromMemory(file.data(), file.size()), "LoadFromMemory");
    ntc::DecompressionStats ds;
    run.check(set.ptr->Decompress(&ds, false), "Decompress");
    {
        std::vector<float> decoded(size_t(in->width) * size_t(in->height) * size_t(in->channels));
        ntc::ReadChannelsParameters r;
        r.page = ntc::TextureDataPage::Output;
        r.mipLevel = 0;
        r.firstChannel = 0;
        r.numChannels = in->channels;
        r.pOutData = reinterpret_cast<unsigned char*>(decoded.data());
        r.addressSpace = ntc::AddressSpace::Host;
        r.width = in->width;
        r.height = in->height;
        r.pixelStride = texel;
        r.rowPitch = texel * size_t(in->width);
        r.channelFormat = ntc::ChannelFormat::FLOAT32;
        r.dstColorSpaces = stored;
        run.check(set.ptr->ReadChannels(r), "ReadChannels");
        sink(user, ORR_NTC_DECODED, 0, decoded.data(), decoded.size() * sizeof(float), 0);
    }

    Stream stream{ctx};
    run.check(ctx->OpenReadOnlyMemory(file.data(), file.size(), &stream.ptr), "OpenReadOnlyMemory");
    Metadata meta{ctx};
    run.check(ctx->CreateTextureSetMetadataFromStream(stream.ptr, &meta.ptr),
              "CreateTextureSetMetadataFromStream");
    ntc::LatentTextureDesc const ld = meta.ptr->GetLatentTextureDesc();
    out->latent_width = ld.width;
    out->latent_height = ld.height;
    out->latent_layers = ld.arraySize;
    out->latent_mips = ld.mipLevels;
    out->latent_features = meta.ptr->GetLatentShape().numFeatures;
    std::vector<unsigned char> rows;
    for (int m = 0; m < ld.mipLevels; ++m) {
        for (int layer = 0; layer < ld.arraySize; ++layer) {
            ntc::LatentTextureFootprint fp;
            run.check(meta.ptr->GetLatentTextureFootprint(m, layer, fp),
                      "GetLatentTextureFootprint");
            if (fp.width != mip_extent(ld.width, m) || fp.height != mip_extent(ld.height, m))
                run.fail("latent mip is not half the size of the one above it");
            if (fp.buffer.compressionType != ntc::CompressionType::None)
                run.fail("latent footprint is compressed");
            size_t const row = size_t(fp.width) * 2;
            uint64_t const at = fp.buffer.rangeInStream.offset;
            if (fp.rowPitch < row ||
                at + fp.rowPitch * uint64_t(fp.height - 1) + row > file.size())
                run.fail("latent footprint outside the file");
            rows.resize(row * size_t(fp.height));
            for (int y = 0; y < fp.height; ++y)
                std::memcpy(rows.data() + row * size_t(y),
                            file.data() + at + fp.rowPitch * uint64_t(y), row);
            sink(user, ORR_NTC_LATENT, m, rows.data(), rows.size(), (unsigned long long)layer);
        }
    }

    ntc::InferenceWeightType const kinds[] = {ntc::InferenceWeightType::GenericInt8,
                                              ntc::InferenceWeightType::GenericFP8,
                                              ntc::InferenceWeightType::CoopVecFP8};
    for (ntc::InferenceWeightType kind : kinds) {
        void const* data = nullptr;
        size_t size = 0;
        size_t converted = 0;
        if (meta.ptr->GetInferenceWeights(kind, &data, &size, &converted) != ntc::Status::Ok ||
            !data || size == 0)
            continue;
        ntc::InferenceData inference;
        if (ctx->MakeInferenceData(meta.ptr, kind, 0, &inference) != ntc::Status::Ok) continue;
        unsigned long long const conv =
            kind == ntc::InferenceWeightType::CoopVecFP8 ? converted : 0;
        sink(user, ORR_NTC_WEIGHTS, int(kind), data, size, conv);
        sink(user, ORR_NTC_CONSTANTS, int(kind), &inference.constants, sizeof inference.constants,
             0);
    }
}

}  // namespace

extern "C" int orr_ntc_compress(OrrNtcApi const* api, OrrNtcInput const* in, OrrNtcResult* out,
                                OrrNtcSink sink, void* user, char* error, size_t error_len) {
    Run run{api, error, error_len};
    try {
        compress(run, in, out, sink, user);
        return 0;
    } catch (Failure const&) {
        return 1;
    } catch (...) {
        std::snprintf(error, error_len, "LibNTC raised an exception");
        return 1;
    }
}

// The PCL Stats calls orr-output makes, over pclstats.h from the Reflex SDK.

#include <stdint.h>
#include <windows.h>

#include <pclstats.h>

PCLSTATS_DEFINE()

extern "C" void orr_pcl_stats_init() { PCLSTATS_INIT(0); }

extern "C" void orr_pcl_stats_shutdown() { PCLSTATS_SHUTDOWN(); }

extern "C" void orr_pcl_stats_marker(uint32_t marker, uint64_t frame_id) {
    PCLSTATS_MARKER(marker, frame_id);
}

// The ping message id is 0 until the first init, and 0 is WM_NULL.
extern "C" bool orr_pcl_stats_is_ping(uint32_t message) {
    return message != 0 && PCLSTATS_IS_PING_MSG_ID(message);
}

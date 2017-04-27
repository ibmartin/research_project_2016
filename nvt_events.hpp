#ifndef _NVT_EVENTS_HPP_
#define _NVT_EVENTS_HPP_

#include <nvToolsExt.h>
#include <string>

nvtxEventAttributes_t get_nvtAttrib(std::string message = "NOT_SET", int color = 0x00000000);

#endif
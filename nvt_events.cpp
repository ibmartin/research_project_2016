#include "nvt_events.hpp"

nvtxEventAttributes_t get_nvtAttrib(std::string message, int color){

	nvtxEventAttributes_t attr = { 0 };
	attr.version = NVTX_VERSION;
	attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	attr.colorType = NVTX_COLOR_ARGB;
	//attr.color = 0xFFFFFF;
	attr.color = color;
	attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
	//message = "NOTSET";
	attr.message.ascii = message.c_str();
	nvtxRangePushEx(&attr);

	return attr;
}
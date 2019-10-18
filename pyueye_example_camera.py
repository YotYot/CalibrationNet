#!/usr/bin/env python

#------------------------------------------------------------------------------
#                 PyuEye example - camera modul
#
# Copyright (c) 2017 by IDS Imaging Development Systems GmbH.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#------------------------------------------------------------------------------

from pyueye import ueye
from pyueye_example_utils import (uEyeException, Rect, get_bits_per_pixel,
                                  ImageBuffer, check)

class Camera:
    def __init__(self, device_id=0):
        self.h_cam = ueye.HIDS(device_id)
        self.img_buffers = []

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, _type, value, traceback):
        self.exit()

    def handle(self):
        return self.h_cam

    def alloc(self, buffer_count=10):
        rect = self.get_aoi()
        bpp = get_bits_per_pixel(self.get_colormode())

        for buff in self.img_buffers:
            check(ueye.is_FreeImageMem(self.h_cam, buff.mem_ptr, buff.mem_id))

        for i in range(buffer_count):
            buff = ImageBuffer()
            ueye.is_AllocImageMem(self.h_cam,
                                  rect.width, rect.height, bpp,
                                  buff.mem_ptr, buff.mem_id)
            
            check(ueye.is_AddToSequence(self.h_cam, buff.mem_ptr, buff.mem_id))

            self.img_buffers.append(buff)

        ueye.is_InitImageQueue(self.h_cam, 0)

    def init(self):
        ret = ueye.is_InitCamera(self.h_cam, None)
        if ret != ueye.IS_SUCCESS:
            self.h_cam = None
            raise uEyeException(ret)
            
        return ret

    def exit(self):
        ret = None
        if self.h_cam is not None:
            ret = ueye.is_ExitCamera(self.h_cam)
        if ret == ueye.IS_SUCCESS:
            self.h_cam = None

    def get_aoi(self):
        rect_aoi = ueye.IS_RECT()
        ueye.is_AOI(self.h_cam, ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi, ueye.sizeof(rect_aoi))

        return Rect(rect_aoi.s32X.value,
                    rect_aoi.s32Y.value,
                    rect_aoi.s32Width.value,
                    rect_aoi.s32Height.value)

    def set_aoi(self, x, y, width, height):
        rect_aoi = ueye.IS_RECT()
        rect_aoi.s32X = ueye.int(x)
        rect_aoi.s32Y = ueye.int(y)
        rect_aoi.s32Width = ueye.int(width)
        rect_aoi.s32Height = ueye.int(height)

        return ueye.is_AOI(self.h_cam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))

    def capture_video(self, wait=False):
        wait_param = ueye.IS_WAIT if wait else ueye.IS_DONT_WAIT
        return ueye.is_CaptureVideo(self.h_cam, wait_param)

    def stop_video(self):
        return ueye.is_StopLiveVideo(self.h_cam, ueye.IS_FORCE_VIDEO_STOP)
    
    def freeze_video(self, wait=False):
        wait_param = ueye.IS_WAIT if wait else ueye.IS_DONT_WAIT
        return ueye.is_FreezeVideo(self.h_cam, wait_param)

    def set_colormode(self, colormode):
        check(ueye.is_SetColorMode(self.h_cam, colormode))
        
    def get_colormode(self):
        ret = ueye.is_SetColorMode(self.h_cam, ueye.IS_GET_COLOR_MODE)
        return ret

    def get_format_list(self):
        count = ueye.UINT()
        check(ueye.is_ImageFormat(self.h_cam, ueye.IMGFRMT_CMD_GET_NUM_ENTRIES, count, ueye.sizeof(count)))
        format_list = ueye.IMAGE_FORMAT_LIST(ueye.IMAGE_FORMAT_INFO * count.value)
        format_list.nSizeOfListEntry = ueye.sizeof(ueye.IMAGE_FORMAT_INFO)
        format_list.nNumListElements = count.value
        check(ueye.is_ImageFormat(self.h_cam, ueye.IMGFRMT_CMD_GET_LIST,
                                  format_list, ueye.sizeof(format_list)))
        return format_list

    def get_exposure(self):
        exposure = ueye.DOUBLE()
        check(ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, exposure, ueye.sizeof(exposure)))
        return exposure.value

    def get_exposure_increment(self):
        increment = ueye.DOUBLE()
        check(ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_INC, increment,
                               ueye.sizeof(increment)))
        return increment.value

    def get_exposure_min(self):
        exposure = ueye.DOUBLE()
        check(
            ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN, exposure, ueye.sizeof(exposure)))
        return exposure.value

    def get_exposure_max(self):
        exposure = ueye.DOUBLE()
        check(
            ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX, exposure, ueye.sizeof(exposure)))
        return exposure.value

    def set_exposure(self, exposure):
        new_exposure = ueye.DOUBLE(exposure)
        check(ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, new_exposure, ueye.sizeof(new_exposure)))

    def get_autoexposure_enabled(self):
        val1 = ueye.DOUBLE()
        val2 = ueye.DOUBLE()
        check(ueye.is_SetAutoParameter(self.h_cam, ueye.IS_GET_ENABLE_AUTO_SHUTTER, val1, val2))
        return val1

    def set_autoexposure_enabled(self, enable_autogain):
        if enable_autogain:
            val1 = ueye.DOUBLE(1.0)
        else:
            val1 = ueye.DOUBLE(0.0)
        val2 = ueye.DOUBLE()
        check(ueye.is_SetAutoParameter(self.h_cam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, val1, val2))

    def get_autogain_enabled(self):
        val1 = ueye.DOUBLE()
        val2 = ueye.DOUBLE()
        check(ueye.is_SetAutoParameter(self.h_cam, ueye.IS_GET_ENABLE_AUTO_GAIN, val1, val2))
        return val1

    def set_autogain_enabled(self, enable_autogain):
        if enable_autogain:
            val1 = ueye.DOUBLE(1.0)
        else:
            val1 = ueye.DOUBLE(0.0)
        val2 = ueye.DOUBLE()
        check(ueye.is_SetAutoParameter(self.h_cam, ueye.IS_SET_ENABLE_AUTO_GAIN, val1, val2))

    def get_gain(self):
        return ueye.is_SetHardwareGain(self.h_cam, ueye.IS_GET_MASTER_GAIN, ueye.IS_IGNORE_PARAMETER,
                                       ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER)

    def set_gain(self, gain):
        new_gain = ueye.INT(gain)
        ueye.is_SetHardwareGain(self.h_cam, new_gain, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER,
                                ueye.IS_IGNORE_PARAMETER)

    def get_rgbGain(self):
        #rgbGain = [7,0,52]
        rgbGain = ueye.INT()
        rgbGain[0] = ueye.is_SetHardwareGain(self.h_cam, ueye.IS_IGNORE_PARAMETER, ueye.IS_GET_RED_GAIN,ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER)
        rgbGain[1] = ueye.is_SetHardwareGain(self.h_cam, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER,ueye.IS_GET_GREEN_GAIN, ueye.IS_IGNORE_PARAMETER)
        rgbGain[2] = ueye.is_SetHardwareGain(self.h_cam, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER, ueye.IS_GET_BLUE_GAIN)
        return rgbGain.value

    def set_rgbGain(self, gain):
        #new_gain = ueye.INT(gain)
        ueye.is_SetHardwareGain(self.h_cam,ueye.IS_IGNORE_PARAMETER, 23, 0,31)

    def get_gamma(self):
        gamma = ueye.UINT()
        check(ueye.is_Gamma(self.h_cam, ueye.IS_GAMMA_CMD_GET, gamma, ueye.sizeof(gamma)))
        return gamma.value

    def set_gamma(self, gamma):
        new_gamma = ueye.UINT(gamma)
        check(ueye.is_Gamma(self.h_cam, ueye.IS_GAMMA_CMD_SET, new_gamma, ueye.sizeof(new_gamma)))

    def get_frameRate(self):
        fps = ueye.DOUBLE()
        check(ueye.is_GetFramesPerSecond (self.h_cam, fps))
        return fps.value

    def set_frameRate(self, fps):
        fps = ueye.DOUBLE(fps)
        new_fps = ueye.DOUBLE()
        check(ueye.is_SetFrameRate(self.h_cam, fps, new_fps))
        return new_fps.value

    def get_pixelClock(self):
        pixelClock = ueye.int()
        check(ueye.is_PixelClock(self.h_cam, ueye.IS_PIXELCLOCK_CMD_GET, pixelClock, ueye.sizeof(pixelClock)))
        return pixelClock.value

    def set_pixelClock(self, pixClk):
        new_pixClk = ueye.UINT(pixClk)
        check(ueye.is_PixelClock(self.h_cam, ueye.IS_PIXELCLOCK_CMD_SET, new_pixClk, ueye.sizeof(new_pixClk)))

    def setRopEffect(self):
        # quick and dirty implementation of left-right mirroring
        check(ueye.is_SetRopEffect (self.h_cam, ueye.IS_SET_ROP_MIRROR_LEFTRIGHT, 1, 0))


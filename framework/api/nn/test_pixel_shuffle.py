# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np
import paddle


def pixel_shuffle_np(x, up_factor, data_format='NCHW'):
    if data_format == "NCHW":
        n, c, h, w = x.shape
        new_shape = (n, c // (up_factor * up_factor), up_factor, up_factor, h, w)
        npresult = np.reshape(x, new_shape)
        npresult = npresult.transpose(0, 1, 4, 2, 5, 3)
        oshape = [n, c // (up_factor * up_factor), h * up_factor, w * up_factor]
        npreslut = np.reshape(npresult, oshape)
        return npreslut
    else:
        n, h, w, c = x.shape
        new_shape = (n, h, w, c // (up_factor * up_factor), up_factor, up_factor)
        npresult = np.reshape(x, new_shape)
        npresult = npresult.transpose(0, 1, 4, 2, 5, 3)
        oshape = [n, h * up_factor, w * up_factor, c // (up_factor * up_factor)]
        npresult = np.reshape(npresult, oshape)
        return npresult


class TestPixelShuffleImperative(unittest.TestCase):
    """
    test paddle.nn.PixelShuffle imperative in different input types and parameters. 
    """
    def setUp(self):
        self.dtype1 = 'float32'
        self.up_factor1 = 3
        self.nchw_shape1 = [2, 9, 4, 4]
        self.nhwc_shape1 = [2, 4, 4,9]

        self.dtype2 = 'float64'
        self.up_factor2 = 4
        self.nchw_shape2 = [4, 1024, 77, 77]
        self.nhwc_shape2 = [4, 77, 77, 1024] 

    def test_nchw1(self):
        # nchw test float32
        x = np.random.rand(*self.nchw_shape1).astype(self.dtype1)
        res = pixel_shuffle_np(x, self.up_factor1, data_format='NCHW')
        x = paddle.to_tensor(x)
        pixel_shuffle = paddle.nn.PixelShuffle(self.up_factor1, data_format='NCHW')
        paddle_res = pixel_shuffle(x).numpy()
        self.assertEqual((paddle_res == res).all(), True)

    def test_nhwc1(self):
        # nhwc test float32
        x = np.random.rand(*self.nhwc_shape1).astype(self.dtype1)
        res = pixel_shuffle_np(x, self.up_factor1, data_format='NHWC')
        x = paddle.to_tensor(x)
        pixel_shuffle = paddle.nn.PixelShuffle(self.up_factor1, data_format='NHWC')
        paddle_res = pixel_shuffle(x).numpy()
        self.assertEqual((paddle_res == res).all(), True)

    def test_nchw2(self):
        # nchw test float64
        x = np.random.rand(*self.nchw_shape2).astype(self.dtype2)
        res = pixel_shuffle_np(x, self.up_factor2, data_format='NCHW')
        x = paddle.to_tensor(x)
        pixel_shuffle = paddle.nn.PixelShuffle(self.up_factor2, data_format='NCHW')
        paddle_res = pixel_shuffle(x).numpy()
        self.assertEqual((paddle_res == res).all(), True)

    def test_nhwc2(self):
        # nhwc test float64
        x = np.random.rand(*self.nhwc_shape2).astype(self.dtype2)
        res = pixel_shuffle_np(x, self.up_factor2, data_format='NHWC')
        x = paddle.to_tensor(x)
        pixel_shuffle = paddle.nn.PixelShuffle(self.up_factor2, data_format='NHWC')
        paddle_res = pixel_shuffle(x).numpy()
        self.assertEqual((paddle_res == res).all(), True)


class TestPixelShuffleError(unittest.TestCase):
    """
    test paddle.nn.PixelShuffle error.
    """

    def test_input_type_errors(self):
        x = paddle.randint(low=0, high=255, shape=[2, 9, 4, 4], dtype='int32')
        data_format = "NCHW"
        up_factor = 3
        self.assertRaises(RuntimeError, paddle.nn.PixelShuffle(up_factor, data_format), x)
    

class TestPixelShuffleParameter(unittest.TestCase):
    """
    test paddle.nn.PixelShuffle initialize parameter.
    """

    def test_factor_error(self):
        up_factor = 3
        data_format = 'NCHW'
        x = paddle.rand((4, 16, 224, 224))
        self.assertRaises(ValueError, paddle.nn.PixelShuffle(up_factor, data_format=data_format), x)

if __name__ == '__main__':
    unittest.main()


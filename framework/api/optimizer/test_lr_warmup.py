#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test lr LinearWarmup case
"""


import paddle

start_lr = 0
end_lr = 0.5
warmup_steps = 20
learning_rate = 0.5
    

def test_warmup_1():
    """
    test warmup base test
    """
    scheduler_1 = paddle.optimizer.lr.LinearWarmup(
        learning_rate=learning_rate, warmup_steps=warmup_steps, start_lr=start_lr, end_lr=end_lr, verbose=False
    )
    for i in range(0, 20):
        exp = start_lr + (end_lr - start_lr) * i / warmup_steps
        assert exp == scheduler_1.get_lr()
        scheduler_1.step()
        
       
def test_warmup_2():
    """
    test step > warmup_step
    """
    last_epoch = warmup_steps + 1
    scheduler_2 = paddle.optimizer.lr.LinearWarmup(
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        start_lr=start_lr, 
        end_lr=end_lr, 
        last_epoch=last_epoch, 
        verbose=False
    )
    assert learning_rate == scheduler_2.get_lr() 


def test_warmup_3():
    """
    test warmup_steps <=0 
    """
    try:
        scheduler_3 = paddle.optimizer.lr.LinearWarmup(
            learning_rate=learning_rate, warmup_steps=0, start_lr=start_lr, end_lr=end_lr, verbose=True
        )
    except AssertionError as error:
        assert """'warmup_steps' must be a positive integer""" in str(error)

#!/usr/bin/python
import main
import sys
import time
import math

import requests
import json
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

def get_llm():
    response = requests.get("http://127.0.0.1:8000")
    data = response.json()
    order = data["message"]
    print(order)
    return order

def TurnLeft():
    cmd.mode = 2
    cmd.gaitType = 2
    cmd.velocity = [0.4, 0] # -1  ~ +1
    cmd.yawSpeed = 2
    cmd.footRaiseHeight = 0.1
    udp.SetSend(cmd)
    udp.Send()
    
def run():
    eval(get_llm())


if __name__ == '__main__':

    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff

    udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)

    cmd = sdk.HighCmd()
    state = sdk.HighState()
    udp.InitCmdData(cmd)

    motiontime = 0
    while True:
        time.sleep(1)
        
        
        motiontime = motiontime + 1

        udp.Recv()
        udp.GetRecv(state)

        cmd.mode = 0      # 0:idle, default stand      1:forced stand     2:walk continuously
        cmd.gaitType = 0
        cmd.speedLevel = 0
        cmd.footRaiseHeight = 0
        cmd.bodyHeight = 0
        cmd.euler = [0, 0, 0]
        cmd.velocity = [0, 0]
        cmd.yawSpeed = 0.0
        cmd.reserve = 0

        main.main()
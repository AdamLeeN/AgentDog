import pyaudio
import wave
from openai import OpenAI
#openai
import os
from llm_utils import SeeWhat
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain.agents import tool
import requests


from langchain import hub
from llm_utils import chat_with_model





# 导入所需的库
config={
    "api_key": "sk-OQyJrrKA7y7g4vdUCbDcAf768bB7468d8153BfD93b37Cf42",
    "base_url": "https://api.adamchatbot.chat/v1"
    }
client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
os.environ["OPENAI_API_KEY"] = config["api_key"]
os.environ["openai_api_base"] = config["base_url"]
llm = OpenAI(temperature=1, callbacks=[SeeWhat()])



# 定义一个包装函数，用于查看提示对转录结果的影响
def transcribe(client, audio_filepath, prompt: str) -> str:
    """给定一个提示，转录音频文件。"""
    # 使用OpenAI的音频转录API，创建一个转录对象
    transcript = client.audio.transcriptions.create(
        file=open(audio_filepath, "rb"),
        model="whisper-1",
        prompt=prompt,
    )
    # 返回转录结果的文本
    return transcript.text


def record_audio(filename, duration, sample_rate=44100, channels=2, chunk=1024):
    """
    录制音频并保存为WAV文件
    :param filename: 保存的WAV文件名
    :param duration: 录制时长（秒）
    :param sample_rate: 采样率
    :param channels: 声道数
    :param chunk: 每个音频块的帧数
    """
    # 初始化pyaudio
    audio = pyaudio.PyAudio()

    # 打开音频流
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)

    print("开始录音...")

    frames = []

    # 录制音频数据
    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("录音结束。")

    # 停止和关闭音频流
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 保存音频数据为WAV文件
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))


@tool
def OBEY(instruction: str = 'turn_left', step: str = '1'):
    """
    你是一只智能机器狗。当你收到用户的指令后执行此函数。
    :param instruction: 必要参数，代表用户对机器狗下达的指令，用字符串类型str表示。默认为turn_left表示向左转。
    :param step: 必要参数，代表需要执行指令的次数，用字符串类型str表示。
    :return: 发送成功或失败的信息，字符串类型。
    """

    payload = {"param": step}
    url = f"http://127.0.0.1:8000/{instruction}"
    headers = {
        'Content-Type': 'application/json'
    }
    try:
        for _ in range(int(step)):
            response = requests.post(url, json=payload, headers=headers)
            order = response.json()['response']
            eval(order)
    
    except Exception as e:
        return f"指令执行失败！{e}"
    return f"指令执行成功！请你反馈给用户！"



def main():
    #录音
    temp = input('按回车开始录音5秒')
    filename = "data/zh.wav"  # 保存的文件名
    duration = 5  # 录制时长（秒）
    record_audio(filename, duration)


    instruction = transcribe(client, r'data/zh.wav', '')
    prompt = hub.pull("hwchase17/react-multi-input-json")
    chat_with_model(prompt, input={'input': instruction}, llm=llm, tools=[OBEY], temperature=0.8)







    
import pyaudio
import wave
#from openai import OpenAI  # 不再使用 OpenAI 的音频 API
import os
from llm_utils import SeeWhat
from langchain_ollama import ChatOllama


from langchain_openai import OpenAI
from langchain.agents import tool
import requests

from langchain import hub
from llm_utils import chat_with_model

# 新增导入 faster_whisper
from faster_whisper import WhisperModel

# 如果仍需要 OpenAI LLM 的话可以保持下面的代码
config = {
    "api_key": "sk-OQyJrrKA7y7g4vdUCbDcAf768bB7468d8153BfD93b37Cf42",
    "base_url": "https://api.adamchatbot.chat/v1"
}
client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
os.environ["OPENAI_API_KEY"] = config["api_key"]
os.environ["openai_api_base"] = config["base_url"]
llm = ChatOllama(
    base_url="http://localhost:11434",
    model = "qwen2.5:3b",
    temperature = 0.01,
    num_predict = 4096,
    # other params ...
)

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


# 使用faster_whisper对音频进行转录
def transcribe(audio_filepath: str, prompt: str) -> str:
    """
    使用faster_whisper进行转录
    :param audio_filepath: 要转录的音频文件路径
    :param prompt: 可以在后续对文本做微调的提示，但faster_whisper本身不支持prompt
    :return: 转录结果的文本
    """

    # 初始化whisper模型
    # 可以根据需要更换为"small", "medium", "large"等模型。这里以"medium"为例。
    model = WhisperModel("medium", device="cpu", compute_type="int8")

    # beam_size 和其它参数可根据需求调整
    segments, info = model.transcribe(audio_filepath, beam_size=5)
    transcript_text = ""
    for segment in segments:
        transcript_text += segment.text

    # 如果需要使用prompt对转录结果做简单过滤或拼接，可以在此处进行
    # 这里暂时直接返回原文本
    return transcript_text.strip()


def main():
    #录音
    temp = input('按回车开始录音5秒')
    filename = "data/zh.wav"  # 保存的文件名
    duration = 5  # 录制时长（秒）
    record_audio(filename, duration)

    instruction = transcribe(r'data/zh.wav', '')
    prompt = hub.pull("hwchase17/react-multi-input-json")
    chat_with_model(prompt, input={'input': instruction}, llm=llm, tools=[OBEY], temperature=0.8)


if __name__ == "__main__":
    main()

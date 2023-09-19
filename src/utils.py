import torchaudio
import json
import numpy as np
import pytube
import torch
import os
import subprocess

def get_yt_link(yt_id: str, start_sec: float=0.00, end_sec: float=None) -> str:

    '''
    Gets the corresponding youtube video segment

    - `yt_id`: youtube video id
    - `start_sec`: start second of the segment
    - `end_sec`: end second of the segment

    Returns youtube url with expected format: `https://www.youtube.com/embed/{yd_id}?start={start_sec}&end={end_sec}`
    '''

    youtube_url = ""

    yt_link_prefix = "https://www.youtube.com/embed"

    get_start_sec = int(start_sec)
    get_end_sec = int(end_sec) if end_sec else None

    youtube_url += f"{yt_link_prefix}/{yt_id}"
    youtube_url += f"?start={get_start_sec}"

    if get_end_sec != None:
        youtube_url += f"&end={end_sec}"
    
    return youtube_url


def get_audio_from_yt(youtube_link: str, save_path: str, return_tensor: bool = True) -> dict:
    #  https://pytube.io/en/latest/
    '''
    Uses pytube and ffmpeg to download audio from youtube_link and download it as a `.wav` file

    - `youtube_link`: link of the youtube video
    - `save_path`: path where you want your audio to be saved
    - `return_tensor`: include output of the audio tensor from `torchaudio.load()` in output dictionary

    Returns: dict of keys 'audio_path' and 'audio_tensor'
    '''

    try:
        yt = pytube.YouTube(youtube_link)
        
    except:
        raise Exception("Unable to open YouTube link")
    
    
    yt_streams = yt.streams

    # collect audio files and adaptive (DASH) files
    # not 100% sure what DASH files are but docs say they're higher quality audio
    filtered_stream = yt_streams.filter(
        only_audio=True,
        adaptive=True)
    # it seems these yt streams have codecs mp4 and opus
    
    audio_stream = filtered_stream.get_audio_only()
    # highest bitrate of mp4 file

    vid_name, file_ext = os.path.splitext(audio_stream.default_filename)

    temp_download_path = f"temp{file_ext}"

    # print(audio_stream, vid_name, file_ext, end=f"\n\t --> {temp_download_path}\n")

    download_path = audio_stream.download(
        output_path=None,
        filename = temp_download_path,
        filename_prefix=None,
    )

    # use ffmpeg to convert mp4 to wav
    ffmpeg_command_args = ["ffmpeg", "-i" , download_path, "-ac", "1", "-y", "-f", "wav", save_path]
    
    completed_process = subprocess.run(ffmpeg_command_args,capture_output=True)
    
    # if error occurred
    if completed_process.returncode != 0:
        cmd = " ".join(completed_process.args)
        ffmpeg_stderr = completed_process.stderr.splitlines()

        # last msg seems like the main error msg
        ffmpeg_error_msg = ffmpeg_stderr[-1]

        raise Exception(f"error with command: \n\t>>{cmd}\n ffmpeg error message: {ffmpeg_error_msg}")
    
    # remove the temporary file that was created
    os.remove(temp_download_path)

    output_dict = {"audio_path": save_path, "audio_tensor": None}
    output_dict['audio_path'] = save_path
    
    # if return_tensor == true, will use torchaudio.load to read from the audio_path and return the audio tensor
    if return_tensor:
        audio_waveform, audio_sr = torchaudio.load(filepath=save_path)

        output_dict['audio_tensor'] = (audio_waveform,audio_sr)

    return output_dict
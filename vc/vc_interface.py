import pathlib
import numpy as np
import torch
import faiss

from .infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono, SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono
from .vc_infer_pipeline import VC
from fairseq import checkpoint_utils
from scipy.signal import resample
from scipy.io import wavfile

device = "cuda:0"  # or cpu
is_half = True  # NVIDIA 20 series and higher GPUs are half-precise with no change in quality

def custom_load_state_dict(model, state_dict):
    """
    モデルの状態辞書をカスタム方式でロードする。
    パラメータの形状が一致しない場合は、適切な形状にリサイズしてからロードする。
    """
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            if own_state[name].shape == param.shape:
                own_state[name].copy_(param)
            else:
                # パラメータの形状が一致しない場合、リサイズしてからコピー
                print(f"Resizing model parameter {name} from {param.shape} to {own_state[name].shape}")
                own_state[name].copy_(param.view_as(own_state[name]))
        else:
            print(f"Skipping unexpected parameter {name} from the checkpoint")

def get_vc(sid):
    weight_root = "vc/models"
    person = f"{weight_root}/{sid}/{sid}.pth"
    global cpt, tgt_sr, version

    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    is_half = True  # または、必要に応じて False に設定

    # モデルのインスタンス化と構成
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

    # チェックポイントからパラメータをロード（カスタム処理を使用）
    custom_load_state_dict(net_g, cpt["weight"])

    net_g.eval().to(device)
    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()

    vc = VC(tgt_sr, device, is_half)
    return vc, net_g

def load_hubert():
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(["vc/models/hubert_base.pt"], suffix="",)
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if (is_half):
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

    return hubert_model

def load_audio(input_audio):
    sr, audio = input_audio
    original_sr = sr
    target_sr = 16000
    original_length = len(audio)
    target_length = int(original_length * (target_sr / original_sr))

    return resample(audio, target_length)

def load_wav(path):
    original_sr, audio = wavfile.read(path)
    audio = audio.astype(np.float32) / 32768.0
    original_length = len(audio)
    target_sr = 16000
    target_length = int(original_length * (target_sr / original_sr))

    return resample(audio, target_length)

def convert_voice(hubert_model, model, net_g, input_audio, vcid, f0_up_key, f0_method):
    sid = 0
    f0_file = None
    index_rate = 1

    file_index = f"vc/models/{vcid}/added.index"
    file_big_npy = f"vc/models/{vcid}/total_fea.npy"

    # INDEXファイルの読み込みを試みる
    try:
        index = faiss.read_index(file_index)
        print(f"{file_index} が正常に読み込まれました。")
    except Exception as e:
        print(f"{file_index} の読み込みに失敗しました。エラー: {e}")
    
    # NPYファイルの読み込みを試みる
    try:
        big_npy = np.load(file_big_npy)
        print(f"{file_big_npy} が正常に読み込まれました。")
    except Exception as e:
        print(f"{file_big_npy} の読み込みに失敗しました。エラー: {e}")    
    
    f0_up_key = int(f0_up_key)
    audio = load_audio(input_audio)
    times = [0, 0, 0]

    if_f0 = cpt.get("f0", 1)
    sr = int(cpt.get("sr", 1).replace("k", "")) * 1000

    filter_radius = 3
    resample_sr = 0
    rms_mix_rate = 1
    protect = 0.33
    return sr, model.pipeline(
        hubert_model,
        net_g,
        sid,
        audio,
        "-",
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        f0_file=f0_file,
    )

def batch_convert(input_dir, output_dir, hubert_model, model, net_g, vcid, f0_up_key, f0_method):
    for wav_path in pathlib.Path(input_dir).glob("*.wav"):
        print(f'Converting {wav_path}')
        audio = wavfile.read(wav_path)
        sr, output_audio = convert_voice(hubert_model, model, net_g, audio, vcid, f0_up_key, f0_method)
        wavfile.write(f'{output_dir}/{wav_path.name}', sr, output_audio)
    print('Done')

def concat_audio(audio_data, silence_duration):
    target_sample_rate = 44100

    resampled_audios = []
    for key in sorted(audio_data.keys()):
        original_rate, original_data = audio_data[key]

        resampled_data = resample(original_data, num=target_sample_rate * len(original_data) // original_rate)
        resampled_audios.append(resampled_data)

    silence_samples = int(target_sample_rate * silence_duration)
    silence_data = np.zeros(silence_samples)
    concatenated_audio = np.concatenate([
        np.hstack((resampled_audio, silence_data)) for resampled_audio in resampled_audios[:-1]] + [resampled_audios[-1]])

    return target_sample_rate, concatenated_audio

if __name__ == '__main__':
    hubert_model = load_hubert()
    vc, net_g = get_vc('simple')

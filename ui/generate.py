import gradio as gr
from vc import vc_interface
from tts import tts_interface
from scripts import download
import os
import json

# load vits model names
with open('tts/models/model_list.json', 'r', encoding="utf-8") as file:
    global lang_dic
    lang_dic = json.load(file)


vc_models = ['No conversion']
# load rvc model names
os.makedirs('vc/models', exist_ok=True)
vc_model_root = 'vc/models'
vc_models.extend([d for d in os.listdir(vc_model_root) if os.path.isdir(os.path.join(vc_model_root, d))])


def lang_change(lang):
    global model
    global speaker_list

    download.get_vits_model(lang_dic[lang])
    with open(f'tts/models/{lang_dic[lang]}_speakers.txt', "r", encoding="utf-8") as file:
        speaker_list = [line.strip() for line in file.readlines()]

    model = tts_interface.load_model(lang_dic[lang])
    return gr.update(choices=speaker_list)


def vc_change(vcid):
    if vcid != 'No conversion':
        global hubert_model, vc, net_g
        hubert_model = vc_interface.load_hubert()
        vc, net_g = vc_interface.get_vc(vcid)


def text2speech(lang, text, sid, vcid, pitch, f0method, length_scale):
    global model, speaker_list, vc, net_g, hubert_model  # ここに 'vc' と 'net_g' を追加
    phonemes, tts_audio = tts_interface.generate_speech(model, lang, text, speaker_list.index(sid), False, length_scale)
    if vcid != 'No conversion':
        if vc is not None and net_g is not None:
            print(f"text2speech - vcid: {vcid}, pitch: {pitch}, f0method: {f0method}")
            return phonemes, vc_interface.convert_voice(hubert_model, vc, net_g, tts_audio, vcid, pitch, f0method)
    return phonemes, tts_audio

def acc2speech(lang, text, sid, vcid, pitch, f0method, length_scale):
    global model, speaker_list, vc, net_g, hubert_model
    _, tts_audio = tts_interface.generate_speech(model, lang, text, speaker_list.index(sid), True, length_scale)
    if vcid != 'No conversion':
        # ログ出力を追加
        print(f"acc2speech - vcid: {vcid}, pitch: {pitch}, f0method: {f0method}")
        return vc_interface.convert_voice(hubert_model, vc, net_g, tts_audio, vcid, pitch, f0method)
    return tts_audio


def save_preset(preset_name: str, lang_dropdown: str, sid: str, vcid: str, pitch: int, f0method: str, speed: float):
    print(f"save_preset called with preset_name={preset_name}, lang_dropdown={lang_dropdown}, sid={sid}, vcid={vcid}, pitch={pitch}, f0method={f0method}, speed={speed}")
    path = 'ui/speaker_presets.json'
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            print("Loaded existing presets successfully.")
    except FileNotFoundError:
        print(f"File {path} not found. Creating a new one.")
        data = {}

    if preset_name in data:
        print(f"Preset name '{preset_name}' already exists. Overwriting...")
    else:
        print(f"Creating new preset with name '{preset_name}'.")

    data[preset_name] = {
        'lang': lang_dropdown,
        'sid': speaker_list.index(sid),  # Make sure speaker_list is accessible
        'vcid': vcid,
        'pitch': pitch,
        'f0method': f0method,
        'speed': speed
    }

    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            print(f"Preset '{preset_name}' saved successfully.")
            return "Preset saved successfully."  # 成功メッセージを返す
    except Exception as e:
        print(f"Failed to save preset '{preset_name}'. Error: {e}")
        return f"Error: Failed to save preset. {e}"  # エラーメッセージを返す

def load_presets():
    try:
        with open('ui/speaker_presets.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        return ["Select a preset"] + list(data.keys())
    except FileNotFoundError:
        return ["Select a preset"]

def apply_preset(preset_name):
    if preset_name == "Select a preset":
        return "", "Select a language", "No conversion", 0, "pm", 1.0
    
    try:
        with open('ui/speaker_presets.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            preset = data[preset_name]
            
            # Ensure speaker list is updated
            lang_change(preset['lang'])
            return preset['lang'], speaker_list[preset['sid']], preset['vcid'], preset['pitch'], preset['f0method'], preset['speed']
    except Exception as e:
        print(f"Failed to apply preset '{preset_name}'. Error: {e}")
        return "", "Select a language", "No conversion", 0, "pm", 1.0

def ui():
    with gr.TabItem('Generate'):
        with gr.Row():
            with gr.Column(scale=2):
                text = gr.Textbox(label="Text", value="こんにちは、世界", lines=8)
                text2speech_bt = gr.Button("Generate From Text", variant="primary")

                phonemes = gr.Textbox(label="Phones", interactive=True, lines=8)
                acc2speech_bt = gr.Button("Generate From Phones", variant="primary")

            with gr.Column():
                lang_dropdown = gr.Dropdown(choices=list(lang_dic.keys()), label="Languages",)
                sid = gr.Dropdown(choices=[], label="Speaker")
                lang_dropdown.change(
                    fn=lang_change,
                    inputs=[lang_dropdown],
                    outputs=sid
                )
                speed = gr.Slider(minimum=0.1, maximum=2, step=0.1, label='Speed', value=1)

                vcid = gr.Dropdown(choices=vc_models, label="Voice Conversion", value='No conversion')
                vcid.change(
                    fn=vc_change,
                    inputs=[vcid]
                )
                with gr.Accordion("VC Setteings", open=False):
                    pitch = gr.Slider(minimum=-12, maximum=12, step=1, label='Pitch', value=0)
                    f0method = gr.Radio(label="Pitch Method pm: speed-oriented, harvest: accuracy-oriented", choices=["pm", "harvest"], value="pm")

                preset_name = gr.Textbox(label="Preset Name", interactive=True)
                save_preset_bt = gr.Button("Save Preset")
                save_preset_message = gr.Label()  # 成功またはエラーメッセージを表示するためのLabel
                
                presets_dropdown = gr.Dropdown(label="Presets", choices=load_presets())
                presets_dropdown.change(fn=apply_preset, inputs=[presets_dropdown], outputs=[lang_dropdown, sid, vcid, pitch, f0method, speed])

                save_preset_bt.click(
                    fn=save_preset,
                    inputs=[preset_name, lang_dropdown, sid, vcid, pitch, f0method, speed],
                    outputs=[save_preset_message]  # Labelを更新するように指定
                )

                
                
        with gr.Row():
            output_audio = gr.Audio(label="Output Audio", type='numpy')
            text2speech_bt.click(
                fn=text2speech,
                inputs=[lang_dropdown, text, sid, vcid, pitch, f0method, speed],
                outputs=[phonemes, output_audio]
            )
            acc2speech_bt.click(
                fn=acc2speech,
                inputs=[lang_dropdown, phonemes, sid, vcid, pitch, f0method, speed],
                outputs=[output_audio]
            )

import os
import json
import soundfile as sf
import librosa
from datasets import load_dataset
from tqdm import tqdm

def save_manifest(dataset, output_json, audio_dir, max_rows=None):
    os.makedirs(audio_dir, exist_ok=True)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        count = 0
        for item in tqdm(dataset):
            if max_rows and count >= max_rows:
                break
                
            # Usually the audio array is in 'audio' feature
            audio_data = item['audio']
            array = audio_data['array']
            sr = audio_data['sampling_rate']
            
            # Resample to 16000 if necessary
            if sr != 16000:
                array = librosa.resample(y=array, orig_sr=sr, target_sr=16000)
                sr = 16000
                
            duration = len(array) / sr
            
            # Save audio file
            audio_path = os.path.join(audio_dir, f"audio_{count}.wav")
            sf.write(audio_path, array, sr)
            
            # Get text (different datasets might use different keys like 'text', 'sentence', or 'transcript')
            text = item.get('text', item.get('sentence', item.get('transcript', '')))
            
            manifest_entry = {
                "audio_filepath": os.path.abspath(audio_path),
                "duration": duration,
                "text": text.strip()
            }
            f.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')
            count += 1
            
    print(f"Saved {count} items to {output_json}")

if __name__ == "__main__":
    print("Preparing Kathbath (Hindi)...")
    try:
        # Kathbath is often gated or large, we'll stream or load a small part for demo, but users can run it fully
        kb_ds = load_dataset("ai4bharat/Kathbath", "hindi", split="train", streaming=True, token=True)
        save_manifest(kb_ds, "hindi_manifest.json", "data/hindi_audio", max_rows=10000) # Remove max_rows for full
    except Exception as e:
        print(f"Error loading Kathbath: {e}")

    print("Preparing Zac Sample Dataset (English)...")
    try:
        zac_ds = load_dataset("canopylabs/zac-sample-dataset", split="train")
        save_manifest(zac_ds, "english_zac_manifest.json", "data/english_zac_audio")
    except Exception as e:
        print(f"Error loading Zac Dataset: {e}")
        
    print("Preparing People's Speech (English)...")
    try:
        # People's speech is very large, loading with streaming
        ps_ds = load_dataset("MLCommons/peoples_speech", "clean", split="train", streaming=True)
        save_manifest(ps_ds, "english_ps_manifest.json", "data/english_ps_audio", max_rows=25000)
    except Exception as e:
        print(f"Error loading People's Speech: {e}")
        
    print("Done! You can now use these JSON manifests in your training config.")

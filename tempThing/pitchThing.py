import librosa
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt

# Ses dosyasını yükle
audio_path = "C:/Users/bahab/Downloads/output_directory/htdemucs/transient/drums.mp3"
y, sr = librosa.load(audio_path, sr=16000)

# Transient'leri tespit et
onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

print(f"Toplam transient sayısı: {len(onset_times)}")
print(f"Transient zamanları (saniye): {onset_times}")

# Spektrogramu hesapla
D = np.abs(librosa.stft(y))
times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr)
freqs = librosa.fft_frequencies(sr=sr)

# Pitch map
pitch_map = {
    36: "Acoustic Bass Drum (Kick)",    # 50-150 Hz
    38: "Acoustic Snare",              # 150-400 Hz
    42: "Closed Hi-Hat",              # 400-600 Hz
    46: "Open Hi-Hat",                # 400-600 Hz
    43: "High Floor Tom",             # 80-200 Hz
    45: "Low Mid Tom",                # 80-200 Hz
    48: "High Mid Tom",               # 80-200 Hz
    50: "High Tom",                   # 80-200 Hz
    49: "Crash Cymbal 1",             # 600-1200 Hz
    57: "Crash Cymbal 2",             # 600-1200 Hz
    51: "Ride Cymbal 1",              # 600-1200 Hz
    55: "Splash Cymbal",              # 600-1200 Hz
    52: "Chinese Cymbal"              # 600-1200 Hz
}

# Minimum pitch oranları (toplam notaların yaklaşık %100’ü)
target_distribution = {
    36: 0.50,  # Kick (ilk çıktıda baskın, ama biraz azaltalım)
    38: 0.15,  # Snare (ekleyelim)
    42: 0.10,  # Closed Hi-Hat (ekleyelim)
    46: 0.05,  # Open Hi-Hat (ekleyelim)
    43: 0.05,  # High Floor Tom
    45: 0.05,  # Low Mid Tom
    48: 0.05,  # High Mid Tom
    50: 0.05,  # High Tom
    49: 0.05,  # Crash Cymbal 1
    57: 0.05,  # Crash Cymbal 2
    51: 0.05,  # Ride Cymbal 1
    55: 0.05,  # Splash Cymbal
    52: 0.05   # Chinese Cymbal
}

# Transient'lere pitch atama
def classify_pitch(freq_content):
    # Frekans aralıklarında enerji hesaplama
    low_freq_energy = np.mean(freq_content[(freqs >= 50) & (freqs <= 150)])      # Kick
    mid_freq_energy = np.mean(freq_content[(freqs >= 150) & (freqs <= 400)])     # Snare
    hihat_freq_energy = np.mean(freq_content[(freqs >= 400) & (freqs <= 600)])   # Hi-Hat
    cymbal_freq_energy = np.mean(freq_content[(freqs >= 600) & (freqs <= 1200)]) # Cymbal
    tom_freq_energy = np.mean(freq_content[(freqs >= 80) & (freqs <= 200)])      # Tom’lar

    # Enerji normalizasyonu (aralık genişliklerine göre)
    energies = {
        "kick": low_freq_energy / (150 - 50),
        "snare": mid_freq_energy / (400 - 150),
        "hihat": hihat_freq_energy / (600 - 400),
        "cymbal": cymbal_freq_energy / (1200 - 600),
        "tom": tom_freq_energy / (200 - 80)
    }
    max_energy = max(energies.values())
    if max_energy == 0:
        return 36  # Enerji yoksa varsayılan kick

    # En yüksek enerjiye göre sınıflandırma
    dominant = max(energies, key=energies.get)
    if dominant == "kick":
        return 36
    elif dominant == "snare":
        return 38
    elif dominant == "hihat":
        return 42 if np.random.random() < 0.7 else 46  # %70 closed, %30 open
    elif dominant == "cymbal":
        cymbal_options = [49, 57, 51, 55, 52]
        return np.random.choice(cymbal_options)
    else:  # Tom
        tom_options = [43, 45, 48, 50]
        return np.random.choice(tom_options)

# Minimum pitch sayısını sağla
def enforce_minimum_pitches(notes, total_notes):
    pitches = [note.pitch for note in notes]
    for pitch, target_ratio in target_distribution.items():
        min_count = int(total_notes * target_ratio)
        current_count = pitches.count(pitch)
        if current_count < min_count:
            # Eksik notaları rastgele ekle
            for _ in range(min_count - current_count):
                idx = np.random.randint(0, len(onset_times))
                start_time = onset_times[idx]
                end_time = start_time + 0.1
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
                notes.append(note)

# MIDI dosyası oluştur
midi = pretty_midi.PrettyMIDI()
drum_instrument = pretty_midi.Instrument(program=0, is_drum=True)

print("Nota eklendi:")
notes = []
for onset_frame in onset_frames:
    onset_idx = onset_frame
    if onset_idx < D.shape[1]:
        freq_content = D[:, onset_idx]
        pitch = classify_pitch(freq_content)
        start_time = onset_times[np.where(onset_frames == onset_frame)[0][0]]
        end_time = start_time + 0.1
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
        drum_instrument.notes.append(note)
        notes.append(note)
        print(f"pitch={pitch}, start={start_time:.3f}, end={end_time:.3f}")

# Minimum pitch oranlarını uygula
enforce_minimum_pitches(drum_instrument.notes, len(onset_frames))

midi.instruments.append(drum_instrument)
midi.write("real_transient_drums_updated_v4.mid")

print(f"MIDI dosyası oluşturuldu: real_transient_drums_updated_v4.mid, Toplam nota: {len(drum_instrument.notes)}")

# Spektrogramu görselleştir
plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='linear', x_axis='time', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title("Spektrogram (Cymbal’lar için kontrol)")
plt.tight_layout()
plt.show()
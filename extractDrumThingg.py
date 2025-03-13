import torch
import torch.nn as nn
import librosa
import pretty_midi
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import random

# Drum pitch aralığı (36-57)
DRUM_PITCHES = [36, 38, 42, 46, 41, 43, 45, 47, 48, 50, 49, 51, 52, 55, 57]
PITCH_MAP = {i: pitch for i, pitch in enumerate(DRUM_PITCHES)}  # 0-14 -> 36-57

class MusicDataset(Dataset):
    def __init__(self, slakh_dir, max_length=160000, hop_length=512):
        self.slakh_dir = slakh_dir
        self.max_length = max_length
        self.hop_length = hop_length
        self.audio_files = []
        self.midi_files = []

        for track in os.listdir(slakh_dir):
            track_path = os.path.join(slakh_dir, track)
            if os.path.isdir(track_path):
                audio_file = os.path.join(track_path, "drum.flac")
                midi_file = os.path.join(track_path, "drum.mid")
                if os.path.exists(audio_file) and os.path.exists(midi_file):
                    self.audio_files.append(audio_file)
                    self.midi_files.append(midi_file)
                    print(f"Found: {audio_file} ve {midi_file}")
                elif os.path.exists(audio_file.replace(".flac", ".wav")):
                    self.audio_files.append(audio_file.replace(".flac", ".wav"))
                    self.midi_files.append(midi_file)
                    print(f"Found: {audio_file.replace('.flac', '.wav')} ve {midi_file}")
                else:
                    print(f"Track {track} için drum dosyaları eksik: {os.listdir(track_path)}")

        print(f"Toplam {len(self.audio_files)} drum dosyası bulundu.")
        if len(self.audio_files) == 0:
            print(f"Hata: {slakh_dir} dizininde drum.flac ve drum.mid dosyaları bulunamadı!")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio, sr = librosa.load(self.audio_files[idx], sr=16000)
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        spec = librosa.stft(audio, hop_length=self.hop_length)
        spec_db = librosa.amplitude_to_db(np.abs(spec))

        midi = pretty_midi.PrettyMIDI(self.midi_files[idx])
        piano_roll = midi.get_piano_roll(fs=sr)[36:58]  # Sadece drum pitch’leri (36-57)
        target_length = spec_db.shape[1]  # Spektrogramın zaman boyutu
        if piano_roll.shape[1] > target_length:
            piano_roll = piano_roll[:, :target_length]
        elif piano_roll.shape[1] < target_length:
            piano_roll = np.pad(piano_roll, ((0, 0), (0, target_length - piano_roll.shape[1])), mode='constant')
        onset_roll = (piano_roll > 0).astype(np.float32)
        frame_roll = onset_roll  # Frame roll’u onset ile aynı tutuyoruz

        return (torch.tensor(spec_db, dtype=torch.float32),
                torch.tensor(onset_roll, dtype=torch.float32),
                torch.tensor(frame_roll, dtype=torch.float32))


class OnsetsAndFrames(nn.Module):
    def __init__(self, num_pitches=15):
        super(OnsetsAndFrames, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 3)),  # Zaman boyutu küçültülür
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding=(2, 0)),  # Frekans boyutunu küçült
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((5, 1)),  # Frekans boyutunu daha da küçült
        )
        self.onset_rnn = nn.LSTM(960, 64, batch_first=True, bidirectional=True, num_layers=2)
        self.frame_rnn = nn.LSTM(960, 64, batch_first=True, bidirectional=True, num_layers=2)
        self.onset_fc = nn.Linear(128, num_pitches)  # 128 = 64 * 2 (bidirectional)
        self.frame_fc = nn.Linear(128, num_pitches)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, freq, time]
        x = self.conv(x)    # [batch, 64, reduced_freq, reduced_time]
        x = x.permute(0, 3, 1, 2)  # [batch, time, channels, freq]
        x = x.reshape(x.size(0), x.size(1), -1)  # [batch, time, features]
        if x.size(2) != 960:
            print(f"Uyarı: Özellik boyutu {x.size(2)} beklenen 960 değil. Padding veya kırpma uygulanıyor.")
            if x.size(2) > 960:
                x = x[:, :, :960]
            else:
                x = torch.nn.functional.pad(x, (0, 960 - x.size(2)))
        onset_out, _ = self.onset_rnn(x)
        frame_out, _ = self.frame_rnn(x)
        onset_pred = self.onset_fc(onset_out)
        frame_pred = self.frame_fc(frame_out)
        return onset_pred, frame_pred


def audio_to_spectrogram(audio_path, sr=16000, max_length=160000, hop_length=512):
    audio, sr = librosa.load(audio_path, sr=sr)
    if len(audio) > max_length:
        audio = audio[:max_length]
    spec = librosa.stft(audio, hop_length=hop_length)
    spec_db = librosa.amplitude_to_db(np.abs(spec))
    return torch.tensor(spec_db, dtype=torch.float32).unsqueeze(0)


def predictions_to_midi(onset_pred, frame_pred, sr=16000, hop_length=512, output_path="drum_notes.mid"):
    onset_pred = torch.sigmoid(onset_pred).squeeze(0).detach().cpu().numpy()
    frame_pred = torch.sigmoid(frame_pred).squeeze(0).detach().cpu().numpy()

    print(f"Onset_pred max: {onset_pred.max()}, min: {onset_pred.min()}")
    print(f"Frame_pred max: {frame_pred.max()}, min: {frame_pred.min()}")
    print(f"Pitch bazında max onset_pred: {np.max(onset_pred, axis=0)}")

    plt.plot(np.max(onset_pred, axis=1))
    plt.title("Onset_pred Maksimum Değerleri (Zaman Ekseni)")
    plt.show()

    midi = pretty_midi.PrettyMIDI()
    drum_instrument = pretty_midi.Instrument(program=0, is_drum=True)

    time_per_frame = hop_length / sr
    last_starts = [-1] * len(DRUM_PITCHES)
    min_counts = {pitch: 0 for pitch in DRUM_PITCHES}

    for t in range(len(onset_pred)):
        onset_frame = onset_pred[t]
        threshold = np.percentile(onset_frame, 90)  # Dinamik eşik (en yüksek %10)
        top_pitches = np.where(onset_frame > threshold)[0]
        start_time = t * time_per_frame
        if len(top_pitches) > 0:
            weights = onset_frame[top_pitches] / np.sum(onset_frame[top_pitches])
            pitch_idx = np.random.choice(top_pitches, p=weights)
            drum_pitch = PITCH_MAP[pitch_idx]
            if last_starts[pitch_idx] == -1 or start_time - last_starts[pitch_idx] > 0.02:
                end_time = start_time + 0.1
                last_starts[pitch_idx] = start_time
                min_counts[drum_pitch] += 1
                note = pretty_midi.Note(velocity=100, pitch=drum_pitch, start=start_time, end=end_time)
                drum_instrument.notes.append(note)
                print(f"Nota eklendi: pitch={drum_pitch}, start={start_time}, end={end_time}")

    total_notes = len(drum_instrument.notes)
    for pitch, count in min_counts.items():
        if count < 5:
            for _ in range(5 - count):
                start_time = np.random.uniform(0, total_notes * time_per_frame / sr)
                end_time = start_time + 0.1
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
                drum_instrument.notes.append(note)
                print(f"Minimum nota eklendi: pitch={pitch}, start={start_time}, end={end_time}")

    midi.instruments.append(drum_instrument)
    midi.write(output_path)
    print(f"MIDI dosyası oluşturuldu: {output_path}, Toplam nota: {len(drum_instrument.notes)}")


def train_model(slakh_dir, epochs=12, hop_length=512):
    dataset = MusicDataset(slakh_dir=slakh_dir, hop_length=hop_length)
    if len(dataset) == 0:
        raise ValueError("Veri setinde hiç dosya bulunamadı, eğitim iptal edildi.")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = OnsetsAndFrames()
    print("Sıfırdan eğitim başlıyor (drum pitch’leri için optimize edildi).")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0 if i < 5 else 2.0 for i in range(15)]))

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for spec, onset_target, frame_target in dataloader:
            optimizer.zero_grad()
            onset_pred, frame_pred = model(spec)  # [batch, time, num_pitches]

            # Hedeflerin boyut sıralamasını düzelt ve zaman boyutunu eşleştir
            onset_target = onset_target.permute(0, 2, 1)  # [batch, time, num_pitches]
            frame_target = frame_target.permute(0, 2, 1)  # [batch, time, num_pitches]

            # Zaman boyutunu modelin çıkışına eşitle
            target_time = onset_pred.shape[1]
            if onset_target.shape[1] > target_time:
                onset_target = onset_target[:, :target_time, :]
                frame_target = frame_target[:, :target_time, :]
            elif onset_target.shape[1] < target_time:
                onset_target = torch.nn.functional.pad(onset_target, (0, 0, 0, target_time - onset_target.shape[1]))
                frame_target = torch.nn.functional.pad(frame_target, (0, 0, 0, target_time - frame_target.shape[1]))

            # Pitch boyutunu eşitle
            onset_target = onset_target[:, :, :15]
            frame_target = frame_target[:, :, :15]

            loss = criterion(onset_pred, onset_target) + 0.5 * criterion(frame_pred, frame_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch {epoch}, Batch Loss: {loss.item()}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Average Loss: {avg_loss}")

        if epoch in [2, 8]:
            print(f"Erken doğrulama (Epoch {epoch})...")
            test_audio = dataset.audio_files[0]
            spec = audio_to_spectrogram(test_audio, hop_length=hop_length)
            model.eval()
            with torch.no_grad():
                onset_pred, frame_pred = model(spec)
                predictions_to_midi(onset_pred, frame_pred, hop_length=hop_length, output_path=f"early_validation_{epoch}.mid")
            model.train()

    output_path = "updated_onsets_frames_drums.pth"
    torch.save(model.state_dict(), output_path)
    print(f"Model kaydedildi: {output_path}")

    print("Son doğrulama yapılıyor...")
    test_audio = dataset.audio_files[0]
    spec = audio_to_spectrogram(test_audio, hop_length=hop_length)
    model.eval()
    with torch.no_grad():
        onset_pred, frame_pred = model(spec)
        predictions_to_midi(onset_pred, frame_pred, hop_length=hop_length, output_path="validation.mid")


def extract_drum_notes(audio_path, model_path="updated_onsets_frames_drums.pth", output_path="drum_notes.mid", hop_length=512):
    model = OnsetsAndFrames()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    spec = audio_to_spectrogram(audio_path, hop_length=hop_length)
    print(f"Spektrogram boyutu: {spec.shape}")

    with torch.no_grad():
        onset_pred, frame_pred = model(spec)
        print(f"Onset tahmin boyutu: {onset_pred.shape}, Frame tahmin boyutu: {frame_pred.shape}")
        predictions_to_midi(onset_pred, frame_pred, hop_length=hop_length, output_path=output_path)


if __name__ == "__main__":
    slakh_dir = 'C:/Users/bahab/OneDrive/Masaüstü/drumsonly/'
    audio_path = "C:/Users/bahab/Downloads/output_directory/htdemucs/transient/drums.mp3"
    model_path = "C:/Users/bahab/PycharmProjects/PythonProject1/updated_onsets_frames_drums.pth"
    hop_length = 512

    train_model(slakh_dir=slakh_dir, epochs=12, hop_length=hop_length)

    try:
        extract_drum_notes(audio_path, model_path=model_path, hop_length=hop_length)
    except PermissionError:
        print("Hata: 'drum_notes.mid' dosyasına yazma izni yok. Farklı bir yol deneniyor...")
        output_path = "C:/Users/bahab/OneDrive/Masaüstü/drum_notes.mid"
        extract_drum_notes(audio_path, model_path=model_path, output_path=output_path, hop_length=hop_length)
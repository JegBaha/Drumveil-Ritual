import torch
import torch.nn as nn
import librosa
import pretty_midi
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import random

class MusicDataset(Dataset):
    def __init__(self, slakh_dir, max_length=160000):
        self.slakh_dir = slakh_dir
        self.max_length = max_length
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
        spec = librosa.stft(audio)
        spec_db = librosa.amplitude_to_db(np.abs(spec))

        midi = pretty_midi.PrettyMIDI(self.midi_files[idx])
        piano_roll = midi.get_piano_roll(fs=sr)[0:88]
        target_length = spec_db.shape[1]
        if piano_roll.shape[1] > target_length:
            piano_roll = piano_roll[:, :target_length]
        onset_roll = (piano_roll > 0).astype(np.float32)
        frame_roll = onset_roll

        return (torch.tensor(spec_db, dtype=torch.float32),
                torch.tensor(onset_roll, dtype=torch.float32),
                torch.tensor(frame_roll, dtype=torch.float32))


class OnsetsAndFrames(nn.Module):
    def __init__(self):
        super(OnsetsAndFrames, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(48, 48, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d((1, 3)),
        )
        self.onset_rnn = None
        self.frame_rnn = None
        self.onset_fc = nn.Linear(176, 88)
        self.frame_fc = nn.Linear(176, 88)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        if self.onset_rnn is None or self.frame_rnn is None:
            input_size = x.size(-1)
            self.onset_rnn = nn.LSTM(input_size, 88, batch_first=True, bidirectional=True)
            self.frame_rnn = nn.LSTM(input_size, 88, batch_first=True, bidirectional=True)
        onset_out, _ = self.onset_rnn(x)
        frame_out, _ = self.frame_rnn(x)
        onset_pred = self.onset_fc(onset_out)
        frame_pred = self.frame_fc(frame_out)
        return onset_pred, frame_pred


def audio_to_spectrogram(audio_path, sr=16000, max_length=160000):
    audio, sr = librosa.load(audio_path, sr=sr)
    if len(audio) > max_length:
        audio = audio[:max_length]
    spec = librosa.stft(audio)
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
    last_starts = [-1] * 88  # Her pitch için ayrı kontrol
    for t in range(len(onset_pred)):
        onset_frame = onset_pred[t]
        threshold = max(0.2, onset_pred.max() * 0.5)  # Dinamik eşik
        top_pitches = np.where(onset_frame > threshold)[0]
        start_time = t * time_per_frame
        if len(top_pitches) > 0:
            weights = onset_frame[top_pitches] / np.sum(onset_frame[top_pitches])  # Ağırlıklar
            pitch = np.random.choice(top_pitches, p=weights)  # Ağırlıklı seçim
            if last_starts[pitch] == -1 or start_time - last_starts[pitch] > 0.02:
                end_time = start_time + 0.1
                last_starts[pitch] = start_time
                if pitch % 15 == 0:
                    drum_pitch = 36  # Kick
                elif pitch % 15 == 1:
                    drum_pitch = 38  # Snare
                elif pitch % 15 == 2:
                    drum_pitch = 42  # Closed Hi-Hat
                elif pitch % 15 == 3:
                    drum_pitch = 46  # Open Hi-Hat
                elif pitch % 15 == 4:
                    drum_pitch = 41  # Low Floor Tom
                elif pitch % 15 == 5:
                    drum_pitch = 43  # High Floor Tom
                elif pitch % 15 == 6:
                    drum_pitch = 45  # Low Mid Tom
                elif pitch % 15 == 7:
                    drum_pitch = 47  # Mid Tom
                elif pitch % 15 == 8:
                    drum_pitch = 48  # High Mid Tom
                elif pitch % 15 == 9:
                    drum_pitch = 50  # High Tom
                elif pitch % 15 == 10:
                    drum_pitch = 49  # Crash Cymbal 1
                elif pitch % 15 == 11:
                    drum_pitch = 51  # Ride Cymbal 1
                elif pitch % 15 == 12:
                    drum_pitch = 52  # Chinese Cymbal
                elif pitch % 15 == 13:
                    drum_pitch = 55  # Splash Cymbal
                else:
                    drum_pitch = 57  # Crash Cymbal 2
                note = pretty_midi.Note(
                    velocity=100, pitch=drum_pitch, start=start_time, end=end_time
                )
                drum_instrument.notes.append(note)
                print(f"Nota eklendi: pitch={drum_pitch}, start={start_time}, end={end_time}")

    midi.instruments.append(drum_instrument)
    midi.write(output_path)
    print(f"MIDI dosyası oluşturuldu: {output_path}, Toplam nota: {len(drum_instrument.notes)}")


def train_model(slakh_dir, epochs=12):
    dataset = MusicDataset(slakh_dir=slakh_dir)
    if len(dataset) == 0:
        raise ValueError("Veri setinde hiç dosya bulunamadı, eğitim iptal edildi.")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = OnsetsAndFrames()
    print("Magenta’dan ilhamla sıfırdan eğitim başlıyor.")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0))

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for spec, onset_target, frame_target in dataloader:
            optimizer.zero_grad()
            onset_pred, frame_pred = model(spec)
            onset_target = onset_target.permute(0, 2, 1)[:, :onset_pred.shape[1], :]
            frame_target = frame_target.permute(0, 2, 1)[:, :frame_pred.shape[1], :]
            loss = criterion(onset_pred, onset_target) + criterion(frame_pred, frame_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Batch - Onset_pred max: {torch.sigmoid(onset_pred).max().item()}, "
                  f"Frame_pred max: {torch.sigmoid(frame_pred).max().item()}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Average Loss: {avg_loss}")

        if epoch == 2:
            print("Erken doğrulama yapılıyor...")
            test_audio = dataset.audio_files[0]
            spec = audio_to_spectrogram(test_audio)
            model.eval()
            with torch.no_grad():
                onset_pred, frame_pred = model(spec)
                predictions_to_midi(onset_pred, frame_pred, output_path="early_validation.mid")
            model.train()

        if epoch == 8:
            print("Erken doğrulama yapılıyor...")
            test_audio = dataset.audio_files[0]
            spec = audio_to_spectrogram(test_audio)
            model.eval()
            with torch.no_grad():
                onset_pred, frame_pred = model(spec)
                predictions_to_midi(onset_pred, frame_pred, output_path="early_validation2.mid")
            model.train()

    output_path = "updated_onsets_frames_drums.pth"
    torch.save(model.state_dict(), output_path)
    print(f"Model kaydedildi: {output_path}")

    print("Doğrulama yapılıyor...")
    test_audio = dataset.audio_files[0]
    spec = audio_to_spectrogram(test_audio)
    model.eval()
    with torch.no_grad():
        onset_pred, frame_pred = model(spec)
        predictions_to_midi(onset_pred, frame_pred, output_path="validation.mid")


def extract_drum_notes(audio_path, model_path="updated_onsets_frames_drums.pth", output_path="drum_notes.mid"):
    model = OnsetsAndFrames()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    spec = audio_to_spectrogram(audio_path)
    print(f"Spektrogram boyutu: {spec.shape}")

    with torch.no_grad():
        onset_pred, frame_pred = model(spec)
        print(f"Onset tahmin boyutu: {onset_pred.shape}, Frame tahmin boyutu: {frame_pred.shape}")
        predictions_to_midi(onset_pred, frame_pred, output_path=output_path)


if __name__ == "__main__":
    slakh_dir = 'C:/Users/bahab/OneDrive/Masaüstü/drumsonly/'
    audio_path = "C:/Users/bahab/Downloads/output_directory/htdemucs/hypnosis/drums.mp3"
    model_path = "C:/Users/bahab/PycharmProjects/PythonProject1/updated_onsets_frames_drums.pth"

    train_model(slakh_dir=slakh_dir, epochs=12)

    try:
        extract_drum_notes(audio_path, model_path=model_path)
    except PermissionError:
        print("Hata: 'drum_notes.mid' dosyasına yazma izni yok. Farklı bir yol deneniyor...")
        output_path = "C:/Users/bahab/OneDrive/Masaüstü/drum_notes.mid"
        extract_drum_notes(audio_path, model_path=model_path, output_path=output_path)
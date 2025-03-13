import torch
import torch.nn as nn
import librosa
import pretty_midi
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import os
import matplotlib.pyplot as plt
import random

# Pitch dağılımı
pitch_counts = {
    36: 50000, 38: 30000, 42: 10000, 43: 5000, 46: 5000, 49: 5000,
    41: 5000, 45: 2000, 47: 2000, 48: 2000, 50: 2000, 51: 2000, 52: 1000, 55: 1000, 57: 1000
}

# Daha agresif düzeltme ağırlıkları
total_counts = sum(pitch_counts.values())
pitch_probabilities = {pitch: count / total_counts for pitch, count in pitch_counts.items()}
correction_weights = {pitch: (1.0 / (prob + 1e-6)) ** 2 for pitch, prob in pitch_probabilities.items()}

class MusicDataset(Dataset):
    def __init__(self, slakh_dir, max_length=160000):
        self.slakh_dir = slakh_dir
        self.max_length = max_length
        self.audio_files = []
        self.midi_files = []

        for track in os.listdir(slakh_dir):
            track_path = os.path.join(slakh_dir, track)
            if os.path.isdir(track_path):
                audio_file = None
                midi_file = None
                for file in os.listdir(track_path):
                    if file.endswith('.flac'):
                        audio_file = os.path.join(track_path, file)
                    elif file.endswith('.mid'):
                        midi_file = os.path.join(track_path, file)
                if audio_file and midi_file:
                    self.audio_files.append(audio_file)
                    self.midi_files.append(midi_file)
                print(f"Found: {audio_file} ve {midi_file}")

        self.total_files = len(self.audio_files)
        print(f"Toplam {self.total_files} drum dosyası çifti bulundu.")
        if self.total_files == 0:
            raise ValueError(f"Hata: {slakh_dir} dizininde drum.flac ve drum.mid dosyaları bulunamadı!")

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
    last_starts = [-1] * 88
    for t in range(len(onset_pred)):
        onset_frame = onset_pred[t]
        pitch_means = np.mean(onset_pred, axis=0)
        thresholds = pitch_means * 0.9
        top_pitches = np.where(onset_frame > [thresholds[p] for p in range(len(onset_frame))])[0]
        start_time = t * time_per_frame
        if len(top_pitches) > 0:
            weights = [onset_frame[p] * correction_weights.get(p, 1.0) for p in top_pitches]
            weights = np.array(weights) / np.sum(weights)
            pitch = np.random.choice(top_pitches, p=weights)
            if last_starts[pitch] == -1 or start_time - last_starts[pitch] > 0.02:
                end_time = start_time + 0.1
                last_starts[pitch] = start_time
                if pitch % 15 == 0:
                    drum_pitch = 36
                elif pitch % 15 == 1:
                    drum_pitch = 38
                elif pitch % 15 == 2:
                    drum_pitch = 42
                elif pitch % 15 == 3:
                    drum_pitch = 46
                elif pitch % 15 == 4:
                    drum_pitch = 41
                elif pitch % 15 == 5:
                    drum_pitch = 43
                elif pitch % 15 == 6:
                    drum_pitch = 45
                elif pitch % 15 == 7:
                    drum_pitch = 47
                elif pitch % 15 == 8:
                    drum_pitch = 48
                elif pitch % 15 == 9:
                    drum_pitch = 50
                elif pitch % 15 == 10:
                    drum_pitch = 49
                elif pitch % 15 == 11:
                    drum_pitch = 51
                elif pitch % 15 == 12:
                    drum_pitch = 52
                elif pitch % 15 == 13:
                    drum_pitch = 55
                else:
                    drum_pitch = 57
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
    if dataset.total_files == 0:
        raise ValueError("Veri setinde hiç dosya bulunamadı, eğitim iptal edildi.")

    # Veri setini train ve test olarak ayır (%80 train, %20 test)
    indices = list(range(dataset.total_files))
    np.random.shuffle(indices)
    split = int(np.floor(0.8 * dataset.total_files))
    train_indices, test_indices = indices[:split], indices[split:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = OnsetsAndFrames()
    print("Magenta’dan ilhamla sıfırdan eğitim başlıyor.")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0))

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for spec, onset_target, frame_target in train_loader:
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
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Average Train Loss: {avg_loss}")

        # Test seti üzerinde doğrulama
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for spec, onset_target, frame_target in test_loader:
                onset_pred, frame_pred = model(spec)
                onset_target = onset_target.permute(0, 2, 1)[:, :onset_pred.shape[1], :]
                frame_target = frame_target.permute(0, 2, 1)[:, :frame_pred.shape[1], :]
                loss = criterion(onset_pred, onset_target) + criterion(frame_pred, frame_target)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch {epoch}, Average Test Loss: {avg_test_loss}")
        model.train()

        if epoch == 3:
            print("Erken doğrulama yapılıyor...")
            test_audio = dataset.audio_files[train_indices[0]]  # Train setinden bir dosya
            spec = audio_to_spectrogram(test_audio)
            model.eval()
            with torch.no_grad():
                onset_pred, frame_pred = model(spec)
                predictions_to_midi(onset_pred, frame_pred, output_path="early_validation.mid")
            model.train()
        if epoch == 8:
            print("Erken doğrulama yapılıyor...")
            test_audio = dataset.audio_files[train_indices[0]]  # Train setinden bir dosya
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
    test_audio = dataset.audio_files[test_indices[0]]  # Test setinden bir dosya
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
    audio_path = "C:/Users/bahab/Downloads/output_directory/htdemucs/transient/drums.mp3"
    model_path = "C:/Users/bahab/PycharmProjects/PythonProject1/updated_onsets_frames_drums.pth"

    train_model(slakh_dir=slakh_dir, epochs=12)

    try:
        extract_drum_notes(audio_path, model_path=model_path)
    except PermissionError:
        print("Hata: 'drum_notes.mid' dosyasına yazma izni yok. Farklı bir yol deneniyor...")
        output_path = "C:/Users/bahab/OneDrive/Masaüstü/drum_notes.mid"
        extract_drum_notes(audio_path, model_path=model_path, output_path=output_path)
import os
import tarfile
import yaml
import shutil

tar_path = 'E:/slakh2100_flac_redux.tar.gz'
output_path = 'C:/Users/bahab/OneDrive/Masaüstü/drumsonly/'

print(f"Creating output directory at: {output_path}")
try:
    os.makedirs(output_path, exist_ok=True)
    print("Output directory created successfully.")
except Exception as e:
    print(f"Failed to create output directory: {e}")
    exit(1)

# Mevcut track'leri kontrol et (dinamik sayı)
existing_tracks = {track for track in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, track))}
print(f"Found {len(existing_tracks)} existing tracks in {output_path}")

print(f"Opening tar.gz file: {tar_path}")
try:
    with tarfile.open(tar_path, 'r:gz') as tar:
        print("Tar.gz file opened successfully.")
        tar_members = {m.name: m for m in tar.getmembers()}  # Sözlük kullanarak hızlandır
        print(f"Total files in tar: {len(tar_members)}")

        # Tüm metadata dosyalarını tara
        metadata_files = [m for m in tar_members.values() if m.name.startswith('slakh2100_flac_redux/train/Track') and m.name.endswith('metadata.yaml')]
        print(f"Total tracks to process: {len(metadata_files)}")
        print(f"Remaining tracks to extract: {len(metadata_files) - len(existing_tracks)}")

        for member in metadata_files:
            track_dir = member.name.split('/')[2]  # Örneğin, Track00001
            # Eğer track zaten çıkarılmışsa atla
            if track_dir in existing_tracks:
                print(f"Skipping already processed track: {track_dir}")
                continue

            print(f"Processing metadata: {member.name}")
            try:
                tar.extract(member, path='temp')
                temp_metadata_path = os.path.join('temp', member.name)
                print(f"Extracted metadata to: {temp_metadata_path}")

                with open(temp_metadata_path, 'r') as f:
                    metadata = yaml.load(f, Loader=yaml.FullLoader)
                print("Metadata loaded successfully.")

                stems = metadata.get('stems', {})
                drum_sources = [s for s_id, s in stems.items() if s.get('is_drum', False)]

                if drum_sources:
                    drum_source = drum_sources[0]
                    source_id = list(stems.keys())[list(stems.values()).index(drum_source)]
                    print(f"Found drum source: {source_id}")

                    audio_file = f'slakh2100_flac_redux/train/{track_dir}/stems/{source_id}.flac'
                    mid_file = f'slakh2100_flac_redux/train/{track_dir}/MIDI/{source_id}.mid'
                    print(f"Looking for audio file: {audio_file}")
                    print(f"Looking for MIDI file: {mid_file}")

                    if audio_file in tar_members:
                        try:
                            tar.extract(audio_file, path='temp')
                            temp_audio_path = os.path.join('temp', audio_file)
                            new_audio_path = os.path.join(output_path, track_dir, 'drum.flac')
                            os.makedirs(os.path.dirname(new_audio_path), exist_ok=True)
                            shutil.move(temp_audio_path, new_audio_path)
                            print(f"Extracted and moved drum audio for {track_dir} to {new_audio_path}")
                        except Exception as e:
                            print(f"Failed to extract/move audio file {audio_file}: {e}")
                    else:
                        print(f"Audio file {audio_file} not found in tar.gz")

                    if mid_file in tar_members:
                        try:
                            tar.extract(mid_file, path='temp')
                            temp_mid_path = os.path.join('temp', mid_file)
                            new_mid_path = os.path.join(output_path, track_dir, 'drum.mid')
                            os.makedirs(os.path.dirname(new_mid_path), exist_ok=True)
                            shutil.move(temp_mid_path, new_mid_path)
                            print(f"Extracted and moved drum MIDI for {track_dir} to {new_mid_path}")
                        except Exception as e:
                            print(f"Failed to extract/move MIDI file {mid_file}: {e}")
                    else:
                        print(f"MIDI file {mid_file} not found in tar.gz")
                else:
                    print(f"No drum sources found in {track_dir}")

                try:
                    os.remove(temp_metadata_path)
                    print(f"Removed temporary metadata file: {temp_metadata_path}")
                except Exception as e:
                    print(f"Failed to remove temporary metadata file {temp_metadata_path}: {e}")

            except Exception as e:
                print(f"Error processing {member.name}: {e}")

except Exception as e:
    print(f"Failed to open tar.gz file {tar_path}: {e}")
    exit(1)

print("Cleaning up temporary directory...")
if os.path.exists('temp'):
    try:
        shutil.rmtree('temp')
        print("Temporary directory cleaned successfully.")
    except Exception as e:
        print(f"Failed to clean temporary directory: {e}")
else:
    print("No temporary directory to clean.")

print("Script completed.")
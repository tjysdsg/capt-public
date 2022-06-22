dir=$1

for file in "$dir"/*.mp3; do
  filename="${file%.*}"
  ffmpeg -y -i "$file" -acodec pcm_s16le -ac 1 -ar 16000 "${filename}.wav"
done

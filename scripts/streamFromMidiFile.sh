#!/bin/bash

# Usage example: `streamFromMidiFile.sh music.midi funny.gif <YT_KEY_FROM_YT_DASHBOARD>`

midifile=$1
video=$2
ytkey=$3

# Stream URL is the same for every account.
yturl=rtmp://a.rtmp.youtube.com/live2/

timidity $midifile -Ow -o - | ffmpeg -i - -i $video -filter_complex '[1:v]scale=1280:720, loop=loop=-1:size=3276:start=0' -map 0:a -map 1:v -fflags +discardcorrupt -f flv $yturl$ytkey


import os
import yt_dlp

def download_audio(youtube_url, output_folder="downloads"):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        print("Download complete!")
    except Exception as e:
        print(f"Error: {e}")

youtube_url = "https://youtu.be/NxEHSAfFlK8?si=jd57V8EmFxbyIJrA"

download_audio(youtube_url)

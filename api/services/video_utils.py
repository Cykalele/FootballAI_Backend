import os
import re
import sys
import subprocess
from urllib.parse import urlencode
import cv2
import requests
import yt_dlp
from bs4 import BeautifulSoup


def is_valid_mp4(file_path: str) -> bool:
    """
    Checks if a file exists, has an MP4 extension, and contains the 'ftyp' header.
    This is a basic MP4 file integrity validation.
    """
    try:
        if not os.path.exists(file_path):
            print(f"❌ File does not exist: {file_path}")
            return False

        file_size = os.path.getsize(file_path)
        print(f"📏 File size: {round(file_size / 1024 / 1024, 2)} MB")

        with open(file_path, 'rb') as f:
            header = f.read(128)
            print(f"🔍 MP4 header: {header[:16]}")
            if b"ftyp" in header and file_path.lower().endswith(".mp4"):
                print("✅ Valid MP4 detected.")
                return True
            else:
                print("❌ Missing 'ftyp' header or incorrect file extension.")
                return False
    except Exception as e:
        print(f"❌ Error during MP4 validation: {e}")
        return False


def compress_video(input_path: str, output_path: str, crf: int = 28, preset: str = "fast") -> bool:
    """
    Compresses a video using ffmpeg with the H.264 codec.
    The CRF value determines compression quality (lower is better quality, larger file size).
    """
    try:
        ffmpeg_command = [
            "ffmpeg", "-i", input_path,
            "-vcodec", "libx264", "-crf", str(crf), "-preset", preset,
            "-y", output_path
        ]
        subprocess.run(ffmpeg_command, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def get_video_fps(video_path: str) -> int:
    """
    Returns the frames per second (FPS) of the given video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps


def download_dropbox_file(dropbox_url: str, output_path: str, crf: int = 28, preset: str = "fast") -> bool:
    """
    Downloads a file from Dropbox, then compresses it using ffmpeg.
    """
    if "dl=0" in dropbox_url:
        dropbox_url = dropbox_url.replace("dl=0", "dl=1")
    elif "dl=1" not in dropbox_url:
        dropbox_url += "&dl=1"

    try:
        temp_download_path = output_path.replace(".mp4", "_original.mp4")
        print("📥 Downloading Dropbox file locally …")
        with requests.get(dropbox_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192

            print(f"📦 Size: {round(total_size / 1024 / 1024, 2)} MB")

            with open(temp_download_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            percent = int((downloaded / total_size) * 100)
                            bar = f"[{'█' * (percent // 2)}{'.' * (50 - percent // 2)}] {percent}%"
                            sys.stdout.write(f"\r📥 Download: {bar}")
                            sys.stdout.flush()
            if total_size > 0:
                print("\n✅ Download complete.")
            else:
                print("✅ Download complete (size unknown).")

        print("🗜️ Compressing downloaded file …")
        if compress_video(temp_download_path, output_path, crf=crf, preset=preset):
            os.remove(temp_download_path)
            print("✅ Compression successful – original file deleted.")
            return True
        else:
            print("❌ Compression failed – keeping original file.")
            return False

    except Exception as e:
        print(f"❌ Error downloading or compressing file: {e}")
        return False


def extract_download_url_from_form_html(html: str) -> str | None:
    """
    Extracts a direct download URL from a <form> element in HTML.
    Used for Google Drive HTML pages that contain a 'download-form'.
    """
    soup = BeautifulSoup(html, "html.parser")
    form = soup.find("form", {"id": "download-form"})
    if not form:
        print("❌ No 'download-form' found in HTML.")
        return None

    action = form.get("action")
    if not action:
        print("❌ No 'action' attribute found in the form.")
        return None

    params = {}
    for input_tag in form.find_all("input"):
        name = input_tag.get("name")
        value = input_tag.get("value", "")
        if name:
            params[name] = value

    full_url = action + "?" + urlencode(params)
    print(f"🔗 Extracted download link from <form>: {full_url}")
    return full_url


def save_response_content(response, destination, chunk_size=32768):
    """
    Saves the content of an HTTP response to a file in chunks.
    """
    total = 0
    try:
        content_type = response.headers.get("Content-Type", "")
        print(f"📄 Response Content-Type: {content_type}")
        if "text/html" in content_type:
            print("❌ Response is HTML – likely an error, not a video file.")
            return False

        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
        print(f"✅ Download complete ({round(total / 1024 / 1024, 2)} MB).")
        return True
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False


def download_google_drive_file(drive_url: str, output_path: str, crf: int = 28, preset: str = "fast") -> bool:
    """
    Downloads a file from Google Drive, handling HTML confirmation pages when necessary,
    then compresses it using ffmpeg.
    """
    try:
        match = re.search(r'/d/([a-zA-Z0-9_-]+)', drive_url)
        if not match:
            print("❌ No valid Google Drive file ID found.")
            return False
        file_id = match.group(1)
        print(f"🔑 Extracted Google Drive ID: {file_id}")

        session = requests.Session()
        base_url = "https://drive.google.com/uc?export=download"

        print("📡 Sending initial file request …")
        with session.get(base_url, params={'id': file_id}, stream=True) as response:
            content_type = response.headers.get("Content-Type", "")
            print(f"📄 Response Content-Type: {content_type}")

            if "text/html" in content_type:
                print("📨 HTML response detected – checking for form …")
                html_text = response.text
                download_url = extract_download_url_from_form_html(html_text)

                if not download_url:
                    print("❌ No download link found – aborting.")
                    return False

                print("📥 Fetching extracted direct link …")
                response = session.get(download_url, stream=True)
            else:
                print("📥 Direct download detected.")

            temp_download_path = output_path.replace(".mp4", "_original.mp4")
            if save_response_content(response, temp_download_path):
                print("🗜️ Compressing downloaded file …")
                if compress_video(temp_download_path, output_path, crf=crf, preset=preset):
                    os.remove(temp_download_path)
                    print("✅ Compression successful – original file deleted.")
                    return True
                else:
                    print("❌ Compression failed – keeping original file.")
                    return False
            else:
                print("❌ Could not save file.")
                return False
    except Exception as e:
        print(f"❌ Error during Google Drive download: {e}")
        return False


def robust_download_video(url: str, output_path: str) -> bool:
    """
    Attempts to download a video from various sources (Dropbox, Google Drive, YouTube, etc.).
    Automatically validates MP4 and compresses if necessary.
    """
    download_success = False

    if "dropbox.com" in url:
        print("🔁 Dropbox link detected. Downloading and compressing …")
        download_success = download_dropbox_file(url, output_path)

    elif "drive.google.com" in url:
        print("🔁 Google Drive link detected. Downloading and compressing …")
        download_success = download_google_drive_file(url, output_path)

    else:
        try:
            print(f"🔁 Attempting download with yt_dlp: {url}")
            ydl_opts = {
                'outtmpl': output_path,
                'quiet': True,
                'format': 'bestvideo+bestaudio/best',
                'merge_output_format': 'mp4',
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            download_success = os.path.exists(output_path)
        except Exception as e:
            print(f"❌ yt_dlp error: {e}")
            return False

        if download_success and is_valid_mp4(output_path):
            print("📦 yt_dlp download successful, starting compression …")
            temp_output = output_path.replace(".mp4", "_compressed.mp4")
            if compress_video(output_path, temp_output):
                os.remove(output_path)
                os.rename(temp_output, output_path)
                print("✅ Compression complete.")
            else:
                print("⚠️ Compression failed – using original file.")

    if download_success and is_valid_mp4(output_path):
        print("✅ Download and MP4 validation successful.")
        return True
    else:
        print("❌ Download failed or invalid MP4 file.")
        print(f"📂 File exists? {os.path.exists(output_path)}")
        if os.path.exists(output_path):
            print(f"📏 File size: {os.path.getsize(output_path)} Bytes")
        return False

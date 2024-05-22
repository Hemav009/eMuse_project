import spotipy
import spotipy.oauth2 as oauth2
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time

auth_manager = SpotifyClientCredentials('35c8f6430ca648f795ee3233bae8d6b0', '8de6a879a9ca461aa07f424a5804484e')
sp = spotipy.Spotify(auth_manager=auth_manager)

def getTrackIDs(playlist_id):
    track_ids = []
    playlist = sp.playlist(playlist_id)
    for item in playlist['tracks']['items']:
        track = item['track']
        track_ids.append(track['id'])
    return track_ids

def getTrackFeatures(id):
    track_info = sp.track(id)
    name = track_info['name']
    album = track_info['album']['name']
    artist = track_info['album']['artists'][0]['name']
    link = track_info['external_urls']['spotify']  
    track_data = [name, album, artist, link]
    return track_data

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
music_dist = {0: "7pS8tMgJgzQ8XSGpOajOqb", 1: "1n6cpWo9ant4WguEo91KZh", 2: "76e5IppLLEY2K7uQTDlhnh",
              3: "5tWTdQxec7qazfIHx0bUBy", 4: "25NsjjHLJz5NfpwA2w9hvc", 5: "0jxeTEE65lep3L2BnxRs48",
              6: "37i9dQZF1DWZdcdjsv83gQ"} 

for emotion_key, playlist_id in music_dist.items():
    track_ids = getTrackIDs(playlist_id)
    track_list = []

    for i in range(len(track_ids)):
        time.sleep(1)
        track_data = getTrackFeatures(track_ids[i])
        track_list.append(track_data)

    emotion_name = emotion_dict[emotion_key]
    df = pd.DataFrame(track_list, columns=['Name', 'Album', 'Artist', 'Spotify Link'])
    df.to_csv(f'songs/{emotion_name.lower()}_with_links.csv', index=False)

    print(f"CSV for {emotion_name} Generated")

import spotipy
from spotipy import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

username = 'undersaver'
scope = 'user-library-read'

auth_manager = SpotifyOAuth(scope=scope)
sp = spotipy.Spotify(auth_manager=auth_manager)
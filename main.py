from threading import Timer

import webbrowser
from flask import Flask, render_template, request

from analysis import analyze_url

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        url = request.form["url"]
        url_type, data = analyze_url(url)
        if url_type == 1:
            return render_template('playlist_analysis.htm', playlist=data[0], recommended=data[1])
        elif url_type == 2:
            return render_template('song_analysis.htm', song_data=data[0])
        elif url_type == 3:
            return render_template('index.htm', error=2)
        else:
            return render_template('index.htm', error=1)
    return render_template('index.htm')


def open_browser():
    webbrowser.open_new('http://127.0.0.1:2000/')


if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(port=2000, debug=False)

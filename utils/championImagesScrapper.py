import requests
from pathlib import Path

# Get patch version from Riot Games API
patch_version = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()[0]
url = "http://ddragon.leagueoflegends.com/cdn/"+patch_version+"/"

# Champion list
champions = list(requests.get(url+"data/en_US/champion.json").json()["data"].keys())

# Download champions images from DDragon
Path("../champion_images/").mkdir(parents=True, exist_ok=True)
for champion in champions:
    print(champion)
    img_url = url+"img/champion/"+champion+".png"
    with open("../champion_images/"+champion+".png", 'wb') as handler:
        handler.write(requests.get(img_url).content)

